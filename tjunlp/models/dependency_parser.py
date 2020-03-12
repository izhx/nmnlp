"""
模块化Biaffine Dependency Parser，便于开展不同实验， 一些代码来自FastNLP.
"""

from typing import Dict, List, Any
from collections import defaultdict, OrderedDict
from overrides import overrides
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from ..core.model import Model
from ..modules.embedding import build_word_embedding, DeepEmbedding
from ..modules.dropout import WordDropout
from ..modules.linear import NonLinear, Bilinear
from ..modules.encoder import build_encoder
from ..modules.util import initial_parameter


def remove_sep(tensors: List[torch.Tensor]):
    for i in range(len(tensors)):
        tensors[i] = tensors[i][:, :-1]
    return tensors


def _mst(scores):
    """
    with some modification to support parser output for MST decoding
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/models/nn.py#L692
    """
    length = scores.shape[0]
    min_score = scores.min() - 1
    eye = np.eye(length)
    scores = scores * (1 - eye) + min_score * eye
    heads = np.argmax(scores, axis=1)
    heads[0] = 0
    tokens = np.arange(1, length)
    roots = np.where(heads[tokens] == 0)[0] + 1
    if len(roots) < 1:
        root_scores = scores[tokens, 0]
        head_scores = scores[tokens, heads[tokens]]
        new_root = tokens[np.argmax(root_scores / head_scores)]
        heads[new_root] = 0
    elif len(roots) > 1:
        root_scores = scores[roots, 0]
        scores[roots, 0] = 0
        new_heads = np.argmax(scores[roots][:, tokens], axis=1) + 1
        new_root = roots[np.argmin(
            scores[roots, new_heads] / root_scores)]
        heads[roots] = new_heads
        heads[new_root] = 0

    edges = defaultdict(set)
    vertices = {0}
    for dep, head in enumerate(heads[tokens]):
        vertices.add(dep + 1)
        edges[head].add(dep + 1)
    for cycle in _find_cycle(vertices, edges):
        dependents = set()
        to_visit = set(cycle)
        while len(to_visit) > 0:
            node = to_visit.pop()
            if node not in dependents:
                dependents.add(node)
                to_visit.update(edges[node])
        cycle = np.array(list(cycle))
        old_heads = heads[cycle]
        old_scores = scores[cycle, old_heads]
        non_heads = np.array(list(dependents))
        scores[np.repeat(cycle, len(non_heads)),
               np.repeat([non_heads], len(cycle), axis=0).flatten()] = min_score
        new_heads = np.argmax(scores[cycle][:, tokens], axis=1) + 1
        new_scores = scores[cycle, new_heads] / old_scores
        change = np.argmax(new_scores)
        changed_cycle = cycle[change]
        old_head = old_heads[change]
        new_head = new_heads[change]
        heads[changed_cycle] = new_head
        edges[new_head].add(changed_cycle)
        edges[old_head].remove(changed_cycle)

    return heads


def _find_cycle(vertices, edges):
    """
    https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
    https://github.com/tdozat/Parser/blob/0739216129cd39d69997d28cbc4133b360ea3934/lib/etc/tarjan.py
    """
    _index = 0
    _stack = []
    _indices = {}
    _lowlinks = {}
    _onstack = defaultdict(lambda: False)
    _SCCs = []  # pylint:disable=invalid-name

    def _strongconnect(v):  # pylint:disable=invalid-name
        nonlocal _index
        _indices[v] = _index
        _lowlinks[v] = _index
        _index += 1
        _stack.append(v)
        _onstack[v] = True

        for w in edges[v]:  # pylint:disable=invalid-name
            if w not in _indices:
                _strongconnect(w)
                _lowlinks[v] = min(_lowlinks[v], _lowlinks[w])
            elif _onstack[w]:
                _lowlinks[v] = min(_lowlinks[v], _indices[w])

        if _lowlinks[v] == _indices[v]:
            SCC = set()  # pylint:disable=invalid-name
            while True:
                w = _stack.pop()  # pylint:disable=invalid-name
                _onstack[w] = False
                SCC.add(w)
                if not w != v:
                    break
            _SCCs.append(SCC)

    for vertex in vertices:
        if vertex not in _indices:
            _strongconnect(vertex)

    return [SCC for SCC in _SCCs if len(SCC) > 1]


class GraphParser(object):
    """
    基于图的parser base class, 支持贪婪解码和最大生成树解码
    """

    @staticmethod
    def greedy_decoder(arc_matrix, mask=None):
        """
        贪心解码方式, 输入图, 输出贪心解码的parsing结果, 不保证合法的构成树
        :param arc_matrix: [batch, seq_len, seq_len] 输入图矩阵
        :param mask: [batch, seq_len] 输入图的padding mask, 有内容的部分为 1, 否则为 0.
            若为 ``None`` 时, 默认为全1向量. Default: ``None``
        :return heads: [batch, seq_len] 每个元素在树中对应的head(parent)预测结果
        """
        _, seq_len, _ = arc_matrix.shape
        matrix = arc_matrix + \
            torch.diag(arc_matrix.new(seq_len).fill_(-np.inf))
        flip_mask = mask.eq(False)
        matrix.masked_fill_(flip_mask.unsqueeze(1), -np.inf)
        _, heads = torch.max(matrix, dim=2)
        if mask is not None:
            heads *= mask.long()
        return heads

    @staticmethod
    def mst_decoder(arc_matrix, mask=None):
        """
        用最大生成树算法, 计算parsing结果, 保证输出合法的树结构
        :param arc_matrix: [batch, seq_len, seq_len] 输入图矩阵
        :param mask: [batch, seq_len] 输入图的padding mask, 有内容的部分为 1, 否则为 0.
            若为 ``None`` 时, 默认为全1向量. Default: ``None``
        :return heads: [batch, seq_len] 每个元素在树中对应的head(parent)预测结果
        """
        batch_size, seq_len, _ = arc_matrix.shape
        matrix = arc_matrix.clone()
        ans = matrix.new_zeros(batch_size, seq_len).long()
        lens = (mask.long()).sum(
            1) if mask is not None else torch.zeros(batch_size) + seq_len
        for i, graph in enumerate(matrix):
            len_i = lens[i]
            ans[i, :len_i] = torch.as_tensor(
                _mst(graph.detach()[:len_i, :len_i].cpu().numpy()), device=ans.device)
        if mask is not None:
            ans *= mask.long()
        return ans

    @staticmethod
    def decode(graph: torch.Tensor, mask: torch.Tensor, greedy: bool = False):
        # if greedy:
        #     graph = GraphParser.greedy_decoder(graph, mask)
        # else:
        #     graph = GraphParser.mst_decoder(graph, mask)
        graph = graph.max(2)[1]
        return graph


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class ScalarMix(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors, ``mixture = gamma * sum(s_k * tensor_k)``
    where ``s = softmax(w)``, with ``w`` and ``gamma`` scalar parameters.

    In addition, if ``do_layer_norm=True`` then apply layer normalization to each tensor
    before weighting.
    """

    def __init__(self, mixture_size: int, do_layer_norm: bool = False) -> None:
        super(ScalarMix, self).__init__()

        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm

        self.scalar_parameters = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor([0.0])) for _ in range(mixture_size)])
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(self, tensors: List[torch.Tensor],  # pylint: disable=arguments-differ
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute a weighted average of the ``tensors``.  The input tensors an be any shape
        with at least two dimensions, but must all be the same shape.

        When ``do_layer_norm=True``, the ``mask`` is required input.  If the ``tensors`` are
        dimensioned  ``(dim_0, ..., dim_{n-1}, dim_n)``, then the ``mask`` is dimensioned
        ``(dim_0, ..., dim_{n-1})``, as in the typical case with ``tensors`` of shape
        ``(batch_size, timesteps, dim)`` and ``mask`` of shape ``(batch_size, timesteps)``.

        When ``do_layer_norm=False`` the ``mask`` is ignored.
        """
        if len(tensors) != self.mixture_size:
            raise Exception("{} tensors were passed, but the module was initialized to "
                            "mix {} tensors.".format(len(tensors), self.mixture_size))

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = torch.sum(
                ((tensor_masked - mean) * broadcast_mask)**2) / num_elements_not_masked
            return (tensor - mean) / torch.sqrt(variance + 1E-12)

        normed_weights = torch.nn.functional.softmax(torch.cat([parameter for parameter
                                                                in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        if not self.do_layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return self.gamma * sum(pieces)

        else:
            mask_float = mask.float()
            broadcast_mask = mask_float.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask_float) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * _do_layer_norm(tensor,
                                                      broadcast_mask, num_elements_not_masked))
            return self.gamma * sum(pieces)


class DependencyParser(Model, GraphParser):
    """
    主Parser，可更换不同的embedding和encoder。
    """

    def __init__(self,
                 criterion,
                 num_words: int,
                 num_rel: int,
                 num_upos: int,
                 word_embedding: Dict[str, Any],
                 other_embedding: Dict[str, Any] = None,
                 encoder: Dict[str, Any] = None,
                 use_mlp: bool = True,
                 transform_dim: int = 0,
                 arc_dim: int = 250,
                 label_dim: int = 50,
                 dropout: float = 0,
                 greedy_infer: bool = False,
                 **kwargs):
        super().__init__(criterion)
        self.word_embedding = build_word_embedding(
            **word_embedding, num_embeddings=num_words)
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
            self.word_transform = lambda x: x

        if 'feature_fusion' in kwargs:  # 多层融合方式
            if kwargs['feature_fusion'] == 'cat':
                self.fusion = lambda x: torch.cat(x, -1)
                feat_dim *= word_embedding['layer_num']
            else:
                self.fusion = ScalarMix(word_embedding['layer_num'])
        else:
            self.fusion = None

        if other_embedding is not None:
            self.other_embedding = DeepEmbedding(num_upos, **other_embedding)
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
                feat_dim, arc_dim+label_dim, nn.LeakyReLU(0.1)), NonLinear(
                    feat_dim, arc_dim+label_dim, nn.LeakyReLU(0.1))])
        else:
            self.mlp = None
            if encoder is None:
                raise ValueError("Encoder and MLP can't be None at same time!")
            if feat_dim != 2 * (arc_dim + label_dim):
                raise ValueError("Wrong arc and label size!")

        self.dropout = nn.Dropout(dropout)
        self.word_dropout = WordDropout(dropout)

        self.arc_classifier = Bilinear(arc_dim, arc_dim, 1)
        self.rel_classifier = Bilinear(label_dim, label_dim, num_rel)
        self.decoder = lambda x, y: self.decode(x, y, greedy_infer)
        self.split_sizes = [arc_dim, label_dim]
        # for calculate metrics precisely
        self.metric_counter = OrderedDict({'arc': 0, 'rel': 0, 'num': 0})

    def forward(self,  # pylint:disable=arguments-differ
                words: torch.Tensor,
                upostag: torch.Tensor,
                mask: torch.Tensor,  # 有词的地方为True
                head: torch.Tensor = None,
                deprel: torch.Tensor = None,
                **kwargs) -> Dict[str, Any]:
        feat = self.word_embedding(words, **kwargs)
        feat = self.word_transform(feat)
        if self.fusion is not None:
            feat = self.fusion(feat)

        if self.other_embedding is not None:
            upostag = self.other_embedding(upostag, **kwargs)
            feat = torch.cat([feat, upostag], dim=2)

        feat = self.word_dropout(feat)
        if self.encoder is not None:
            feat = self.encoder(feat, **kwargs)  # unpack会去掉[SEP]那一列
            if feat.shape[1] == words.shape[1] - 1:
                mask, head, deprel = remove_sep([mask, head, deprel])

        feat = self.word_dropout(feat)
        if self.mlp is not None:
            feat = [self.word_dropout(self.mlp[i](feat)) for i in range(2)]
            feat = list(feat[0].split(self.split_sizes, dim=2)) + \
                list(feat[1].split(self.split_sizes, dim=2))
        else:
            feat = list(feat.split(self.split_sizes*2, dim=2))

        arc_pred = self.arc_classifier(feat[0], feat[2]).squeeze(-1)  # (b,s,s)
        rel_pred = self.rel_classifier(feat[1], feat[3])  # (b,s,s,c)

        # use gold or predicted arc to predict label
        head_pred = head if self.training else self.decoder(arc_pred, mask)

        rel_pred = torch.gather(rel_pred, 2, head_pred.unsqueeze(
            2).unsqueeze(3).expand(-1, -1, -1, rel_pred.shape[-1])).squeeze(2)
        output = {'head_pred': head_pred, 'rel_pred': rel_pred,
                  'loss': torch.zeros(1)}

        if self.training or self.evaluating:
            loss = self.loss(arc_pred, rel_pred, head, deprel, mask)
            output['loss'] = loss
        if not self.training:
            with torch.no_grad():
                output['metric'] = self.get_metric(
                    head_pred, rel_pred, head, deprel, mask)

        return output

    @overrides
    def get_metric(self,  # pylint:disable=arguments-differ
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
        rel_pred_correct = (
            rel_pred == rel_gt).long() * head_pred_correct
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
