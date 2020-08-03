"""
模块化Biaffine Dependency Parser，便于开展不同实验.
"""

from typing import Dict, List, Any
from collections import OrderedDict
# from itertools import chain

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy

from ..core import Model, Vocabulary
from ..modules.encoder import build_encoder
from ..modules.dropout import WordDropout
from ..modules.linear import NonLinear, Biaffine
from ..modules.util import initial_parameter
from ..nn.chu_liu_edmonds import batch_decode_head
from .embedding import build_word_embedding, DeepEmbedding


def remove_sep(tensors: List[torch.Tensor]):
    for i in range(len(tensors)):
        tensors[i] = tensors[i][:, :-1]
    return tensors


def loss(arc_logits: torch.Tensor, rel_logits: torch.Tensor, arc_gt: torch.Tensor,
         rel_gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    flip_mask = mask.eq(0)
    flip_mask[:, 0] = True

    def one_loss(logits, gt):
        return cross_entropy(logits.view(-1, logits.size(-1)), gt.masked_fill(
            flip_mask, -1).reshape(-1), ignore_index=-1)

    arc_loss = one_loss(arc_logits, arc_gt)
    rel_loss = one_loss(rel_logits, rel_gt)

    return arc_loss + rel_loss


class DependencyParser(Model):
    """
    主Parser，可更换不同的embedding和encoder。
    """

    def __init__(self,
                 criterion,
                 vocab: Vocabulary,
                 word_embedding: Dict[str, Any],
                 transform_dim: int = 0,
                 other_embedding: Dict[str, Any] = None,
                 encoder: Dict[str, Any] = None,
                 use_mlp: bool = True,
                 arc_dim: int = 150,
                 label_dim: int = 150,
                 dropout: float = 0,
                 greedy_infer: bool = False):
        super().__init__(criterion)
        self.word_embedding = build_word_embedding(
            num_embeddings=len(vocab['words']), vocab=vocab, **word_embedding)
        feat_dim: int = self.word_embedding.output_dim
        if transform_dim > 0:
            self.word_transform = NonLinear(feat_dim, transform_dim)
            feat_dim: int = transform_dim
        else:
            self.word_transform = None

        if other_embedding is not None:
            self.other_embedding = DeepEmbedding(len(vocab['upostag']), **other_embedding)
            feat_dim += self.other_embedding.output_dim
        else:
            self.other_embedding = None

        if encoder is not None:
            self.encoder = build_encoder(feat_dim, dropout=dropout, **encoder)
            feat_dim = self.encoder.output_dim
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

        self.arc_classifier = Biaffine(arc_dim, arc_dim, 1)
        self.rel_classifier = Biaffine(label_dim, label_dim, len(vocab['deprel']))
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
                mask: torch.Tensor,  # 有词的地方为1
                head: torch.Tensor = None,
                deprel: torch.Tensor = None,
                seq_lens: torch.Tensor = None,
                **kwargs) -> Dict[str, Any]:
        feat = self.word_embedding(words, mask=mask, **kwargs)
        if self.word_transform is not None:
            feat = self.word_transform(feat)

        if self.other_embedding is not None:
            upostag = self.other_embedding(upostag, **kwargs)
            feat = torch.cat([feat, upostag], dim=2)

        feat = self.word_dropout(feat)
        if self.encoder is not None:
            feat = self.encoder(feat, seq_lens, **kwargs)  # unpack会去掉[SEP]那一列
            if feat.shape[1] == words.shape[1] - 1:
                mask, head, deprel = remove_sep([mask, head, deprel])

        feat = self.word_dropout(feat)
        if self.mlp is not None:
            feat = [self.word_dropout(self.mlp[i](feat)) for i in range(2)]
            feat = list(feat[0].split(self.split_sizes, dim=2)) + list(
                feat[1].split(self.split_sizes, dim=2))
            # feat = list(chain(*map(lambda f: f.split(self.split_sizes, dim=2), feat)))
        else:
            feat = list(feat.split(self.split_sizes * 2, dim=2))

        arc_pred = self.arc_classifier(feat[0], feat[2]).squeeze(-1)  # (b,s,s)
        rel_pred = self.rel_classifier(feat[1], feat[3])  # (b,s,s,c)

        # use gold or predicted arc to predict label
        head_pred = head if self.training else self.decoder(arc_pred, seq_lens)

        rel_pred = torch.gather(rel_pred, 2, head_pred.unsqueeze(
            2).unsqueeze(3).expand(-1, -1, -1, rel_pred.shape[-1])).squeeze(2)
        output = {'head_pred': head_pred, 'rel_pred': rel_pred, 'loss': torch.zeros(1)}

        if head is not None:
            output['loss'] = loss(arc_pred, rel_pred, head, deprel, mask)
            if not self.training:
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

    @staticmethod
    def is_best(metric: Dict[str, float], former: Dict[str, float]) -> bool:
        if metric['UAS'] > former['UAS']:
            return True
        elif metric['UAS'] == former['UAS']:
            return metric['LAS'] > former['LAS']
        else:
            return False
