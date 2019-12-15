"""
code form stanfordnlp
"""

from typing import Dict, Any
from collections import OrderedDict
from overrides import overrides

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence,\
    pack_sequence, PackedSequence

from tjunlp.common.checks import ConfigurationError
from tjunlp.core.model import Model
from tjunlp.core.vocabulary import Vocabulary, DEFAULT_PADDING_INDEX
from tjunlp.models.common.lstm import PackedLSTM, HighwayLSTM
from tjunlp.models.dependency_parser import seq_len_to_mask

from tjunlp.models.chuliu_edmonds import chuliu_edmonds_one_root


def tensor_unsort(sorted_tensor, origin_index):
    """
    Unsort a sorted tensor on its 0-th dimension, based on the original idx.
    """
    assert sorted_tensor.size(0) == len(
        origin_index), "Number of list elements must match with original indices."
    back_index = [x[0]
                  for x in sorted(enumerate(origin_index), key=lambda x: x[1])]
    return sorted_tensor[back_index]


class CharacterModel(nn.Module):
    def __init__(self, vocab, emb_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float, rec_dropout: bool, pad=False, bidirectional=False,
                 attention=True):
        super().__init__()
        # self.args = args
        self.pad = pad
        self.num_dir = 2 if bidirectional else 1
        self.attn = attention
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # char embeddings
        self.char_emb = nn.Embedding(
            len(vocab['char']), emb_dim, padding_idx=0)
        if self.attn:
            self.char_attn = nn.Linear(
                self.num_dir * hidden_dim, 1, bias=False)
            self.char_attn.weight.data.zero_()

        # modules
        self.charlstm = PackedLSTM(emb_dim, hidden_dim,
                                   num_layers, batch_first=True,
                                   dropout=0 if num_layers == 1 else dropout,
                                   rec_dropout=rec_dropout, bidirectional=bidirectional)
        self.char_lstm_h_init = nn.Parameter(  # todo
            torch.zeros(self.num_dir * num_layers, 1, hidden_dim), requires_grad=True)
        self.char_lstm_c_init = nn.Parameter(
            torch.zeros(self.num_dir * num_layers, 1, hidden_dim), requires_grad=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, chars, chars_mask, word_orig_idx, sent_lens, word_lens):
        embs = self.dropout(self.char_emb(chars))
        batch_size = embs.size(0)
        embs = pack_padded_sequence(embs, word_lens, batch_first=True)
        output = self.charlstm(embs, word_lens, hx=(
            self.char_lstm_h_init.expand(self.num_dir * self.num_layers, batch_size,
                                         self.hidden_dim).contiguous(),
            self.char_lstm_c_init.expand(self.num_dir * self.num_layers, batch_size,
                                         self.hidden_dim).contiguous()))

        # apply attention, otherwise take final states
        if self.attn:
            char_reps = output[0]
            weights = torch.sigmoid(
                self.char_attn(self.dropout(char_reps.data)))
            char_reps = PackedSequence(
                char_reps.data * weights, char_reps.batch_sizes)
            char_reps, _ = pad_packed_sequence(char_reps, batch_first=True)
            res = char_reps.sum(1)
        else:
            h, c = output[1]
            res = h[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

        # recover character order and word separation
        res = tensor_unsort(res, word_orig_idx)
        res = pack_sequence(res.split(sent_lens))
        if self.pad:
            res = pad_packed_sequence(res, batch_first=True)[0]

        return res


class WordDropout(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, replacement=None):
        if not self.training or self.drop_prob == 0:
            return x

        mask_size = [y for y in x.size()]
        mask_size[-1] = 1
        drop_mask = torch.rand(*mask_size, device=x.device) < self.drop_prob

        res = x.masked_fill(drop_mask, 0)
        if replacement is not None:
            res = res + drop_mask.float() * replacement

        return res


class PairwiseBilinear(nn.Module):
    """A bilinear module that deals with broadcasting for efficient memory usage.
    Input: tensors of sizes (N x L1 x D1) and (N x L2 x D2)
    Output: tensor of size (N x L1 x L2 x O)"""

    def __init__(self, input1_size, input2_size, output_size, bias=True):
        super().__init__()

        self.input1_size = input1_size
        self.input2_size = input2_size
        self.output_size = output_size

        self.weight = nn.Parameter(torch.zeros(input1_size, input2_size, output_size),
                                   requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(output_size),
                                 requires_grad=True) if bias else 0

    def forward(self, input1, input2):
        n, l1, d1, _, l2, d2, o = * \
            list(input1.size()), *list(input2.size()), self.output_size

        # ((N x L1) x D1) * (D1 x (D2 x O)) -> (N x L1) x (D2 x O)
        temp = torch.mm(input1.view(-1, d1), self.weight.view(-1, d2 * o))
        # (N x L2 x D2) -> (N x D2 x L2)
        input2 = input2.transpose(1, 2)
        # (N L1 d2 O) -> (N L1 O d2)
        temp = temp.view(n, l1, d2, o).transpose(2, 3)
        # (N x (L1 x O) x D2) * (N x D2 x L2) -> (N x (L1 x O) x L2)
        output = temp.reshape(n, l1*o, d2).bmm(input2)
        # (N x (L1 x O) x L2) -> (N x L1 x L2 x O)
        output = output.view(n, l1, o, l2).transpose(2, 3)

        return output


class BiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, output_size, pairwise=True):
        super().__init__()
        if pairwise:
            self.W_bilin = PairwiseBilinear(
                input1_size + 1, input2_size + 1, output_size)
        else:
            self.W_bilin = nn.Bilinear(
                input1_size + 1, input2_size + 1, output_size)

        self.W_bilin.weight.data.zero_()
        self.W_bilin.bias.data.zero_()

    def forward(self, input1, input2):
        input1 = torch.cat([input1, input1.new_ones(
            *input1.size()[:-1], 1)], len(input1.size()) - 1)
        input2 = torch.cat([input2, input2.new_ones(
            *input2.size()[:-1], 1)], len(input2.size()) - 1)
        return self.W_bilin(input1, input2)


class DeepBiaffineScorer(nn.Module):
    def __init__(self, input1_size, input2_size, hidden_size, output_size,
                 hidden_func=F.relu, dropout: float = 0):
        super().__init__()
        self.W1 = nn.Linear(input1_size, hidden_size)
        self.W2 = nn.Linear(input2_size, hidden_size)
        self.hidden_func = hidden_func
        self.scorer = BiaffineScorer(hidden_size, hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input1, input2):
        return self.scorer(self.dropout(self.hidden_func(self.W1(input1))),
                           self.dropout(self.hidden_func(self.W2(input2))))


class Parser(Model):
    """
    code form stanfordnlp.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self,
                 vocab: Vocabulary,
                 criterion: Any,
                 emb_matrix=None,
                 num_layers: int = 3,
                 hidden_dim: int = 100,
                 deep_biaffine_hidden_dim: int = 100,
                 word_emb_dim: int = 75,
                 tag_emb_dim: int = 50,
                 transformed_dim: int = 125,
                 char_model_cfg: Dict = None,
                 pretrained_emb: bool = True,
                 linearization: bool = False,
                 distance: bool = False,
                 dropout: float = 0.5,
                 word_dropout: float = 0.33,
                 rec_dropout: float = 0,
                 **kwargs):
        super().__init__(criterion)
        self.vocab = vocab
        self.kwargs = kwargs
        # self.share_hid = share_hid
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.word_emb_dim = word_emb_dim
        self.tag_emb_dim = tag_emb_dim
        self.unsaved_modules = []
        self.len_words = len(vocab['words'])
        self.len_lemma = len(vocab['lemma'])
        self.len_upos = len(vocab['upos'])
        self.len_deprel = len(vocab['deprel'])

        # input layers
        input_size = 0
        if word_emb_dim > 0:
            # frequent word embeddings
            self.word_emb = nn.Embedding(self.len_words, word_emb_dim,
                                         padding_idx=DEFAULT_PADDING_INDEX)
            self.lemma_emb = nn.Embedding(self.len_lemma, word_emb_dim,
                                          padding_idx=DEFAULT_PADDING_INDEX)
            input_size += word_emb_dim * 2

        if tag_emb_dim > 0:
            self.upos_emb = nn.Embedding(self.len_upos, tag_emb_dim,
                                         padding_idx=DEFAULT_PADDING_INDEX)
            input_size += tag_emb_dim

        if char_model_cfg:
            if not char_model_cfg['char_emb_dim'] > 0:
                raise ConfigurationError('char_emb_dim must greater than 0')
            self.char_model = CharacterModel(
                vocab=vocab, dropout=dropout, **char_model_cfg)
            self.trans_char = nn.Linear(
                char_model_cfg['char_hidden_dim'], transformed_dim, bias=False)
            input_size += transformed_dim
        else:
            self.char_model = None

        if pretrained_emb:  # TODO(izhx): 这里留出 bert elmo 的接口
            # pretrained embeddings, by default this won't be saved into model file
            self.pretrained_emb = nn.Embedding.from_pretrained(
                emb_matrix, freeze=True)
            self.trans_pretrained = nn.Linear(
                emb_matrix.shape[1], transformed_dim, bias=False)
            input_size += transformed_dim
        else:
            self.pretrained_emb = None

        # recurrent layers
        self.highway_lstm = HighwayLSTM(input_size, hidden_dim, num_layers,
                                        batch_first=True, bidirectional=True,
                                        dropout=dropout,
                                        rec_dropout=rec_dropout,
                                        highway_func=torch.tanh)
        self.drop_replacement = nn.Parameter(torch.randn(
            input_size) / np.sqrt(input_size), requires_grad=True)
        self.lstm_h = nn.Parameter(torch.zeros(2 * num_layers, 1, hidden_dim),
                                   requires_grad=True)
        self.lstm_c = nn.Parameter(torch.zeros(2 * num_layers, 1, hidden_dim),
                                   requires_grad=True)

        # classifiers
        self.unlabeled = DeepBiaffineScorer(2 * hidden_dim, 2 * hidden_dim,
                                            deep_biaffine_hidden_dim, 1,
                                            dropout=dropout)
        self.deprel = DeepBiaffineScorer(2 * hidden_dim, 2 * hidden_dim,
                                         deep_biaffine_hidden_dim, self.len_deprel,
                                         dropout=dropout)
        self.linearization = DeepBiaffineScorer(2 * hidden_dim, 2 * hidden_dim,
                                                deep_biaffine_hidden_dim, 1,
                                                dropout=dropout) if linearization else linearization
        self.distance = DeepBiaffineScorer(2 * hidden_dim, 2 * hidden_dim,
                                           deep_biaffine_hidden_dim, 1,
                                           dropout=dropout) if distance else distance

        self.drop = nn.Dropout(dropout)
        self.word_drop = WordDropout(word_dropout)
        self.metrics_counter = OrderedDict({'arc': 0, 'label': 0, 'sample': 0})

    def forward(self, words, upos, pretrained, lemma, heads, deprel, word_ids,
                seq_lens, word_mask=None, wordchars=None, wordchars_mask=None,
                word_lens=None):
        def pack(x):
            return pack_padded_sequence(x, seq_lens, batch_first=True)

        inputs = []
        if self.pretrained_emb:
            pretrained_emb = self.pretrained_emb(pretrained)
            pretrained_emb = self.trans_pretrained(pretrained_emb)
            pretrained_emb = pack(pretrained_emb)
            inputs.append(pretrained_emb)

        if self.word_emb_dim > 0:
            word_emb = self.word_emb(words)
            word_emb = pack(word_emb)
            lemma_emb = self.lemma_emb(lemma)
            lemma_emb = pack(lemma_emb)
            inputs += [word_emb, lemma_emb]

        if self.tag_emb_dim > 0:
            pos_emb = self.upos_emb(upos)
            inputs.append(pack(pos_emb))

        if self.char_model:
            char_reps = self.char_model(
                wordchars, wordchars_mask, word_ids, seq_lens, word_lens)
            char_reps = PackedSequence(self.trans_char(
                self.drop(char_reps.data)), char_reps.batch_sizes)
            inputs.append(char_reps)

        lstm_inputs = torch.cat([x.data for x in inputs], 1)

        lstm_inputs = self.word_drop(lstm_inputs, self.drop_replacement)
        lstm_inputs = self.drop(lstm_inputs)

        lstm_inputs = PackedSequence(lstm_inputs, inputs[0].batch_sizes)

        lstm_outputs, _ = self.highway_lstm(lstm_inputs, seq_lens, hx=(
            self.lstm_h.expand(2 * self.num_layers,
                               words.size(0), self.hidden_dim).contiguous(),
            self.lstm_c.expand(2 * self.num_layers, words.size(0), self.hidden_dim).contiguous()))
        lstm_outputs, _ = pad_packed_sequence(lstm_outputs, batch_first=True)

        unlabeled_scores = self.unlabeled(
            self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
        deprel_scores = self.deprel(
            self.drop(lstm_outputs), self.drop(lstm_outputs))

        if self.linearization or self.distance:
            head_offset = torch.arange(
                words.size(1), device=heads.device).view(1, 1, -1).expand(words.size(0), -1, -1)
            head_offset -= torch.arange(
                words.size(1), device=heads.device).view(1, -1, 1).expand(words.size(0), -1, -1)

        if self.linearization:
            lin_scores = self.linearization(
                self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
            unlabeled_scores += F.logsigmoid(lin_scores *
                                             torch.sign(head_offset).float()).detach()

        if self.distance:
            dist_scores = self.distance(
                self.drop(lstm_outputs), self.drop(lstm_outputs)).squeeze(3)
            dist_pred = 1 + F.softplus(dist_scores)
            dist_target = torch.abs(head_offset)
            dist_kld = -torch.log((dist_target.float() -
                                   dist_pred) ** 2 / 2 + 1)
            unlabeled_scores += dist_kld.detach()

        diag = torch.eye(heads.size(-1), dtype=torch.bool,
                         device=heads.device).unsqueeze(0)
        unlabeled_scores.masked_fill_(diag, -float('inf'))  # 惩罚，不预测自己 -float('inf')

        preds = []

        if self.training or self.evaluating:
            unlabeled_target = heads.masked_fill(word_mask, -1)[:, 1:]
            deprel_target = deprel.masked_fill(word_mask, -1)[:, 1:]

            if self.evaluating:
                unlabeled = F.log_softmax(
                    unlabeled_scores, 2).detach().cpu().numpy()
                deprels = deprel_scores.max(3)[1].detach().cpu().numpy()
                head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in
                             zip(unlabeled, seq_lens)]  # remove attachment for the root
                deprel_seqs = [[deprels[i][j + 1][h]
                                for j, h in enumerate(hs)] for i, hs in enumerate(head_seqs)]
                deprel_pred = torch.zeros_like(deprel)
                head_pred = torch.zeros_like(heads)
                for i, l in enumerate(seq_lens):
                    # pylint: disable=not-callable
                    head_pred[i][1:l] = torch.tensor(
                        head_seqs[i], dtype=torch.int64)
                    deprel_pred[i][1:l] = torch.tensor(
                        deprel_seqs[i], dtype=torch.int64)

                metric = self.get_metrics(
                    head_pred, heads, deprel_pred, deprel)
            else:
                metric = None

            unlabeled_scores = unlabeled_scores.masked_fill(  # 不预测mask  -float('inf')
                word_mask.unsqueeze(1), -float('inf'))[:, 1:, :]  # exclude attachment for the root symbol

            loss = self.criterion(unlabeled_scores.reshape(
                -1, unlabeled_scores.shape[-1]), unlabeled_target.reshape(-1))

            deprel_scores = torch.gather(deprel_scores, 2, heads.unsqueeze(
                2).unsqueeze(3).expand(-1, -1, -1, self.len_deprel))[:, 1:, :, :]
            # shape(B,seq,seq,num) 相当于选真值head那个分支的预测结果(B,seq,head,num)
            deprel_scores = deprel_scores.reshape(-1, self.len_deprel)

            loss += self.criterion(deprel_scores, deprel_target.reshape(-1))

            if self.linearization:
                # lin_scores = lin_scores[:, 1:].masked_select(goldmask)
                lin_scores = torch.gather(
                    lin_scores[:, 1:], 2, heads.unsqueeze(2)).view(-1)
                lin_scores = torch.cat(
                    [-lin_scores.unsqueeze(1) / 2, lin_scores.unsqueeze(1) / 2], 1)
                # lin_target = (head_offset[:, 1:] > 0).long().masked_select(goldmask)
                lin_target = torch.gather(
                    (head_offset[:, 1:] > 0).long(), 2, heads.unsqueeze(2))
                loss += self.criterion(lin_scores.contiguous(),
                                       lin_target.view(-1))

            if self.distance:
                # dist_kld = dist_kld[:, 1:].masked_select(goldmask)
                dist_kld = torch.gather(dist_kld[:, 1:], 2, heads.unsqueeze(2))
                loss -= dist_kld.sum()

            loss /= sum(seq_lens)  # number of words
        else:
            loss = 0
            unlabeled = F.log_softmax(
                unlabeled_scores, 2).detach().cpu().numpy()
            deprels = deprel_scores.max(3)[1].detach().cpu().numpy()

            head_seqs = [chuliu_edmonds_one_root(adj[:l, :l])[1:] for adj, l in
                         zip(unlabeled, seq_lens)]  # remove attachment for the root
            deprel_seqs = [self.vocab['deprel'].unmap([deprels[i][j + 1][h] for j, h in enumerate(hs)]) for i, hs in
                           enumerate(head_seqs)]

            pred_tokens = [[[str(head_seqs[i][j]), deprel_seqs[i][j]] for j in range(seq_lens[i] - 1)] for i in
                           range(len(seq_lens))]
            preds.append(head_seqs)
            preds.append(deprel_seqs)
            preds.append(pred_tokens)

            metric = None

        return {'loss': loss, 'pred': preds, 'metric': metric}

    @overrides
    def get_metrics(self,
                    head_pred: torch.Tensor = None,
                    label_pred: torch.Tensor = None,
                    head_gt: torch.Tensor = None,
                    label_gt: torch.Tensor = None,
                    seq_lens: torch.Tensor = None,
                    reset: bool = False) -> Dict[str, float]:
        """
        Evaluate the performance of prediction.
        reset = False， 计数，返回单次结果， True 用计数计算并清空
        传进来的tree带有虚根
        """
        if reset:
            arc, label, sample = self.metrics_counter.values()
            for k in self.metrics_counter:
                self.metrics_counter[k] = 0
            return {'UAS': arc * 1.0 / sample, 'LAS': label * 1.0 / sample}

        if len(label_pred.shape) > len(label_gt.shape):
            pred_dim, indices_dim = 2, 1
            label_pred = label_pred.max(pred_dim)[indices_dim]

        if seq_lens is None:
            seq_mask = head_pred.new_ones(head_pred.size(), dtype=torch.long)
        else:
            seq_mask = seq_len_to_mask(seq_lens).long()
        # mask out <root> tag
        seq_mask[:, 0] = 0
        head_pred_correct = (head_pred == head_gt).long() * seq_mask
        label_pred_correct = (
            label_pred == label_gt).long() * head_pred_correct
        arc = head_pred_correct.sum().item()
        label = label_pred_correct.sum().item()
        sample = seq_mask.sum().item()
        self.metrics_counter['arc'] += arc
        self.metrics_counter['label'] += label
        self.metrics_counter['sample'] += sample

        return {'UAS': arc * 1.0 / sample, 'LAS': label * 1.0 / sample}

    def is_best(self, metric: Dict[str, float], former: Dict[str, float]) -> bool:
        if metric['UAS'] > former['UAS']:
            return True
        elif metric['UAS'] == former['UAS']:
            return metric['LAS'] > former['LAS']
        else:
            return False


"""
  char_model_cfg:
    emb_dim: 50
    hidden_dim: 50
    num_layers: 3
    rec_dropout: false
    bidirectional: false
    attention: true
"""
