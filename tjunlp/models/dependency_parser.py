"""
模块化Biaffine Dependency Parser，便于开展不同实验， 一些代码来自FastNLP.
"""

from typing import Dict, List, Any
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core import Model, Vocabulary
from ..modules.embedding import build_word_embedding, DeepEmbedding
from ..modules.encoder import build_encoder
from ..modules.fusion import Fusion
from ..modules.dropout import WordDropout
from ..modules.linear import NonLinear, Bilinear
from ..modules.util import initial_parameter
from ..nn.chu_liu_edmonds import batch_decode_head


def remove_sep(tensors: List[torch.Tensor]):
    for i in range(len(tensors)):
        tensors[i] = tensors[i][:, :-1]
    return tensors


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class DependencyParser(Model):
    """
    主Parser，可更换不同的embedding和encoder。
    """

    def __init__(self,
                 criterion,
                 vocab: Vocabulary,
                 word_embedding: Dict[str, Any],
                 other_embedding: Dict[str, Any] = None,
                 encoder: Dict[str, Any] = None,
                 use_mlp: bool = True,
                 transform_dim: int = 0,
                 arc_dim: int = 250,
                 label_dim: int = 50,
                 dropout: float = 0,
                 greedy_infer: bool = True,
                 **kwargs):
        super().__init__(criterion)
        self.word_embedding = build_word_embedding(
            num_embeddings=len(vocab['words']), vocab=vocab, **word_embedding)
        if transform_dim > 0:
            if 'layer_num' in word_embedding and word_embedding['layer_num'] > 1:
                self.word_mlp = nn.ModuleList([NonLinear(
                    self.word_embedding.output_dim, transform_dim, activation=GELU(
                    )) for _ in range(word_embedding['layer_num'])])
                self.word_transform = lambda x: [
                    self.word_mlp[i](x[i]) for i in range(len(x))]
            else:
                self.word_mlp = NonLinear(self.word_embedding.output_dim,
                                          transform_dim, activation=GELU())
                self.word_transform = lambda x: self.word_mlp(x)
            feat_dim: int = transform_dim
        else:
            feat_dim: int = self.word_embedding.output_dim
            self.word_mlp = None
            self.word_transform = None

        try:  # bert 多层融合方式
            method = kwargs['layer_fusion']
            self.fusion = Fusion(method, word_embedding['layer_num'] if method == 'mix' else -1)
            feat_dim *= word_embedding['layer_num'] if method == 'cat' else 1
        except:
            self.fusion = None

        if other_embedding is not None:
            self.other_embedding = DeepEmbedding(len(vocab['upostag']), **other_embedding)
            feat_dim += self.other_embedding.output_dim
        else:
            self.other_embedding = None

        if encoder is not None:
            self.encoder = build_encoder(feat_dim, dropout=dropout, **encoder)
            feat_dim = self.encoder.output_dim
            initial_parameter(self.encoder, initial_method='orthogonal')
        else:
            self.encoder = None

        if use_mlp:
            self.mlp = nn.ModuleList([NonLinear(
                feat_dim, arc_dim + label_dim, nn.LeakyReLU(0.1)), NonLinear(
                feat_dim, arc_dim + label_dim, nn.LeakyReLU(0.1))])
        else:
            self.mlp = None
            if encoder is None:
                raise ValueError("Encoder and MLP can't be None at same time!")
            if feat_dim != 2 * (arc_dim + label_dim):
                raise ValueError("Wrong arc and label size!")

        self.dropout = nn.Dropout(dropout)
        self.word_dropout = WordDropout(dropout)

        self.arc_classifier = Bilinear(arc_dim, arc_dim, 1)
        self.rel_classifier = Bilinear(label_dim, label_dim, len(vocab['deprel']))
        if greedy_infer:
            self.decoder = lambda x, y: x.max(dim=2)[1]
        else:
            self.decoder = batch_decode_head
        self.split_sizes = [arc_dim, label_dim]
        # for calculate metrics precisely
        self.metric_counter = OrderedDict({'arc': 0, 'rel': 0, 'num': 0})

    def forward(self,  # pylint:disable=arguments-differ
                words: torch.Tensor,
                upostag: torch.Tensor,
                mask: torch.Tensor,  # 有词的地方为True
                head: torch.Tensor = None,
                deprel: torch.Tensor = None,
                seq_lens: torch.Tensor = None,
                **kwargs) -> Dict[str, Any]:
        feat = self.word_embedding(words, **kwargs)
        if self.word_transform is not None:
            feat = self.word_transform(feat)
        if self.fusion is not None:
            feat = self.fusion(feat)

        if self.other_embedding is not None:
            upostag = self.other_embedding(upostag, **kwargs)
            feat = torch.cat([feat, upostag], dim=2)

        feat = self.word_dropout(feat)
        if self.encoder is not None:
            feat = self.encoder(feat, seq_lens)  # unpack会去掉[SEP]那一列
            if feat.shape[1] == words.shape[1] - 1:
                mask, head, deprel = remove_sep([mask, head, deprel])

        feat = self.word_dropout(feat)
        if self.mlp is not None:
            feat = [self.word_dropout(self.mlp[i](feat)) for i in range(2)]
            feat = list(feat[0].split(self.split_sizes, dim=2)) + \
                   list(feat[1].split(self.split_sizes, dim=2))
        else:
            feat = list(feat.split(self.split_sizes * 2, dim=2))

        arc_pred = self.arc_classifier(feat[0], feat[2]).squeeze(-1)  # (b,s,s)
        rel_pred = self.rel_classifier(feat[1], feat[3])  # (b,s,s,c)

        # use gold or predicted arc to predict label
        head_pred = head if self.training else self.decoder(arc_pred, seq_lens)

        rel_pred = torch.gather(rel_pred, 2, head_pred.unsqueeze(
            2).unsqueeze(3).expand(-1, -1, -1, rel_pred.shape[-1])).squeeze(2)
        output = {'head_pred': head_pred, 'rel_pred': rel_pred,
                  'loss': torch.zeros(1)}

        if self.training or self.evaluating:
            loss = self.loss(arc_pred, rel_pred, head, deprel, mask)
            output['loss'] = loss
        if not self.training:
            with torch.no_grad():
                output['metric'] = self.get_metrics(
                    head_pred, rel_pred, head, deprel, mask)

        return output

    def get_metrics(self,  # pylint:disable=arguments-differ
                    head_pred: torch.Tensor = None,
                    rel_pred: torch.Tensor = None,
                    head_gt: torch.Tensor = None,
                    rel_gt: torch.Tensor = None,
                    mask: torch.Tensor = None,
                    reset: bool = False,
                    counter: OrderedDict = None) -> Dict[str, float]:
        """
        Evaluate the performance of prediction.
        reset = False， 计数，返回单次结果， True 用计数计算并清空
        """
        if counter is not None:
            arc, rel, num = counter.values()
            return {'UAS': arc * 1.0 / num, 'LAS': rel * 1.0 / num}
        if reset:
            arc, rel, num = self.metric_counter.values()
            for k in self.metric_counter:
                self.metric_counter[k] = 0
            return {'UAS': arc * 1.0 / num, 'LAS': rel * 1.0 / num}

        if len(rel_pred.shape) > len(rel_gt.shape):
            pred_dim, indices_dim = 2, 1
            rel_pred = rel_pred.max(pred_dim)[indices_dim]

        mask = mask.long()
        mask[:, 0] = 0  # mask out <root> tag
        head_pred_correct = (head_pred == head_gt).long() * mask
        rel_pred_correct = (rel_pred == rel_gt).long() * head_pred_correct
        arc = head_pred_correct.sum().item()
        rel = rel_pred_correct.sum().item()
        num = mask.sum().item()
        self.metric_counter['arc'] += arc
        self.metric_counter['rel'] += rel
        self.metric_counter['num'] += num

        return {'UAS': arc * 1.0 / num, 'LAS': rel * 1.0 / num}

    def loss(self, arc_logits: torch.Tensor, rel_logits: torch.Tensor,
             arc_gt: torch.Tensor, rel_gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _, s, n = rel_logits.shape
        flip_mask = mask.eq(False)
        flip_mask[:, 0] = True
        arc_logits = arc_logits.masked_fill(
            flip_mask.unsqueeze(-1).expand(arc_logits.shape), -float('inf'))
        arc_gt = arc_gt.masked_fill(flip_mask, -1)
        arc_loss = F.cross_entropy(
            arc_logits.view(-1, s), arc_gt.reshape(-1), ignore_index=-1)

        rel_gt = rel_gt.masked_fill(flip_mask, -1)
        rel_loss = F.cross_entropy(
            rel_logits.view(-1, n), rel_gt.reshape(-1), ignore_index=-1)

        return arc_loss + rel_loss

    @staticmethod
    def is_best(metric: Dict[str, float], former: Dict[str, float]) -> bool:
        if metric['UAS'] > former['UAS']:
            return True
        elif metric['UAS'] == former['UAS']:
            return metric['LAS'] > former['LAS']
        else:
            return False
