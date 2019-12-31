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
import torch.nn.init as init


from ..core.model import Model
from ..modules.embedding import build_word_embedding, DeepEmbedding
from ..modules.dropout import TimestepDropout
from ..modules.encoder import build_encoder
from ..modules.util import initial_parameter


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
    def decode(graph, mask, greedy: bool = False):
        if greedy:
            graph = GraphParser.greedy_decoder(graph, mask)
        else:
            graph = GraphParser.mst_decoder(graph, mask)
        return graph


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class NonLinear(nn.Module):
    """
    a
    """

    def __init__(self, input_size, hidden_size, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_features=input_size,
                                out_features=hidden_size)
        init.orthogonal_(self.linear.weight.data)
        init.zeros_(self.linear.bias.data)
        if activation is None:
            self._activate = lambda x: x
        else:
            if not callable(activation):
                raise ValueError(
                    f"activation must be callable: type={type(activation)}")
            self._activate = activation

    def forward(self, x):  # pylint:disable=arguments-differ
        x = self.linear(x)
        return self._activate(x)


class Bilinear(nn.Module):
    """
    A bilinear module.
    Input: tensors of sizes (b x n1 x d1) and (b x n2 x d2)
    Output: tensor of size (b x n1 x n2 x O)
    """

    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(
            in1_features, in2_features * out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.out_features = out_features

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:  # pylint:disable=arguments-differ
        b, n1, _, _, n2, d2, o = * \
            list(input1.shape), *list(input2.shape), self.out_features
        # (b, n1, d1) * (d1, o*d2) -> (b, n1, o*d2) -> (b, n1*o, d2)
        lin = input1.matmul(self.weight).reshape(b, n1*o, d2)
        # (b, n1*o, d2) * (b, d2, n2) -> (b, n1*o, n2)
        output = lin.bmm(input2.transpose(1, 2))
        # (b, n1*O, n2) -> (b, n1, n2, O)
        output = output.view(b, n1, o, n2).transpose(2, 3)

        return output  # einsum will cause cuda out of memory, fuck


class DependencyParser(Model, GraphParser):
    """
    主Parser，可更换不同的embedding和encoder。
    """

    def __init__(self,
                 criterion,
                 num_rel: int,
                 num_upos: int,
                 word_embedding: Dict[str, Any],
                 other_embedding: Dict[str, Any] = None,
                 encoder: Dict[str, Any] = None,
                 use_mlp: bool = True,
                 transform_dim: int = 200,
                 arc_dim: int = 250,
                 label_dim: int = 50,
                 dropout: float = 0,
                 use_greedy_infer: bool = False,
                 **kwargs):
        super().__init__(criterion)
        self.word_embedding = build_word_embedding(**word_embedding)
        if transform_dim > 0:
            self.word_mlp = NonLinear(self.word_embedding.output_dim,
                                      transform_dim, activation=GELU())
            feat_dim: int = transform_dim
        else:
            feat_dim: int = self.word_embedding.output_dim
            self.word_mlp = None

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
        self.use_mlp = use_mlp
        if use_mlp:
            self.arc_mlp = NonLinear(
                feat_dim, arc_dim+label_dim, nn.LeakyReLU(0.1))
            self.rel_mlp = NonLinear(
                feat_dim, arc_dim+label_dim, nn.LeakyReLU(0.1))
        else:
            if encoder is None:
                raise ValueError("Encoder and MLP can't be None at same time!")
            if feat_dim != 2 * (arc_dim + label_dim):
                raise ValueError("Wrong arc and label size!")

        self.arc_classifier = Bilinear(arc_dim, arc_dim, 1)
        self.rel_classifier = Bilinear(label_dim, label_dim, num_rel)
        self.split_sizes = [arc_dim, label_dim]
        self.use_greedy_infer = use_greedy_infer
        # for calculate metrics precisely
        self.metrics_counter = OrderedDict({'arc': 0, 'rel': 0, 'sample': 0})

    def forward(self,  # pylint:disable=arguments-differ
                words: torch.Tensor,
                upos: torch.Tensor,
                mask: torch.Tensor,  # 有词的地方为True
                heads: torch.Tensor = None,
                deprel: torch.Tensor = None,
                **kwargs) -> Dict[str, Any]:
        def remove_sep(tensors: List[torch.Tensor]):
            for i in range(len(tensors)):
                tensors[i] = tensors[i][:, :-1]
            return tensors
        feat = self.word_embedding(words, **kwargs)
        if self.word_mlp is not None:
            feat = self.word_mlp(feat)

        if self.other_embedding is not None:
            upos = self.other_embedding(upos, **kwargs)
            feat = torch.cat([feat, upos], dim=2)
        if self.encoder is not None:
            feat = self.encoder(feat, **kwargs)  # unpack会去掉[SEP]那一列
            if feat.shape[1] == words.shape[1] - 1:
                mask, heads, deprel = remove_sep([mask, heads, deprel])
        if self.use_mlp:
            feat = (self.arc_mlp(feat), self.rel_mlp(feat))
            feat = list(feat[0].split(self.split_sizes, dim=2)) + \
                list(feat[1].split(self.split_sizes, dim=2))
        else:
            feat = list(feat.split(self.split_sizes*2, dim=2))

        arc_pred = self.arc_classifier(feat[0], feat[2]).squeeze(-1)  # (b,s,s)

        # use gold or predicted arc to predict label
        if self.training:
            if heads is None:  # 一般不可能
                heads = self.decode(arc_pred, mask, self.use_greedy_infer)
            head_pred = heads
        else:
            head_pred = self.decode(arc_pred, mask, self.use_greedy_infer)

        rel_pred = self.rel_classifier(feat[1], feat[3])  # (b,s,s,c)
        rel_pred = torch.gather(rel_pred, 2, heads.unsqueeze(
            2).unsqueeze(3).expand(-1, -1, -1, rel_pred.shape[-1])).squeeze(2)
        output = dict()

        if self.training or self.evaluating:
            loss = self.loss(arc_pred, rel_pred, heads, deprel, mask)
            output['loss'] = loss
        if self.evaluating:
            with torch.no_grad():
                output['metric'] = self.get_metrics(
                    head_pred, rel_pred, heads, deprel, mask)

        return output

    @overrides
    def get_metrics(self,  # pylint:disable=arguments-differ
                    head_pred: torch.Tensor = None,
                    rel_pred: torch.Tensor = None,
                    head_gt: torch.Tensor = None,
                    rel_gt: torch.Tensor = None,
                    mask: torch.Tensor = None,
                    reset: bool = False) -> Dict[str, float]:
        """
        Evaluate the performance of prediction.
        reset = False， 计数，返回单次结果， True 用计数计算并清空
        """
        if reset:
            arc, rel, sample = self.metrics_counter.values()
            for k in self.metrics_counter:
                self.metrics_counter[k] = 0
            return {'UAS': arc * 1.0 / sample, 'LAS': rel * 1.0 / sample}

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
        sample = mask.sum().item()
        self.metrics_counter['arc'] += arc
        self.metrics_counter['rel'] += rel
        self.metrics_counter['sample'] += sample

        return {'UAS': arc * 1.0 / sample, 'LAS': rel * 1.0 / sample}

    def loss(self, arc_logits: torch.Tensor,
             rel_logits: torch.Tensor,
             arc_gt: torch.Tensor,
             rel_gt: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        b, s, n = rel_logits.shape
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
