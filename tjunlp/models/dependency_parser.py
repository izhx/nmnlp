"""
模块化Biaffine Dependency Parser，便于开展不同实验， 一些代码来自FastNLP.
"""

from typing import Dict, List, Any
from collections import defaultdict, OrderedDict
from overrides import overrides

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ArcBiaffine(nn.Module):
    """
    Biaffine Dependency Parser 的子模块, 用于构建预测边的图
    """

    def __init__(self, hidden_size: int, bias: bool = True):
        """
        :param hidden_size: 输入的特征维度
        :param bias: 是否使用bias. Default: ``True``
        """
        super(ArcBiaffine, self).__init__()
        self.U = nn.Parameter(torch.randn(  # pylint:disable=invalid-name
            hidden_size, hidden_size), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(
            hidden_size), requires_grad=True) if bias else None
        initial_parameter(self)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:  # pylint:disable=arguments-differ
        """
        :param x1: arc-head tensor [batch, length, hidden]
        :param x2: arc-dependent tensor [batch, length, hidden]
        :return : tensor [bacth, length, length]
        """
        # (b, s, h) = (b, s, h) * (h, h)
        x2 = x2.matmul(self.U)
        # (b, s, s) = (b, s, h) * (b, h, s)
        x2 = x2.bmm(x1.transpose(-1, -2))
        if self.bias is not None:
            x2 += x1.matmul(self.bias).unsqueeze(1)
        return x2


class LabelBilinear(nn.Module):
    """
    Biaffine Dependency Parser 的子模块, 用于构建预测边类别的图
    """

    def __init__(self, in1_dim: int, in2_dim: int, num_label: int, bias: bool = True):
        """
        :param in1_dim: 输入的特征1维度
        :param in2_dim: 输入的特征2维度
        :param num_label: 边类别的个数
        :param bias: 是否使用bias. Default: ``True``
        """
        super(LabelBilinear, self).__init__()
        self.bilinear = nn.Bilinear(in1_dim, in2_dim, num_label, bias=bias)
        self.linear = nn.Linear(in1_dim + in2_dim, num_label, bias=False)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:  # pylint:disable=arguments-differ
        """
        :param x1: [batch, seq_len, hidden] 输入特征1, 即label-head
        :param x2: [batch, seq_len, hidden] 输入特征2, 即label-dep
        :return output: [batch, seq_len, num_cls] 每个元素对应类别的概率图
        """
        output = self.bilinear(x1, x2)
        output += self.linear(torch.cat([x1, x2], dim=2))
        return output


class DependencyParser(Model, GraphParser):
    """
    主Parser，可更换不同的embedding和encoder。
    """

    def __init__(self,
                 criterion,
                 num_label: int,
                 num_upos: int,
                 word_embedding: Dict[str, Any],
                 other_embedding: Dict[str, Any] = None,
                 encoder: Dict[str, Any] = None,
                 use_mlp: bool = True,
                 transform_dim: int = 100,
                 arc_dim: int = 250,
                 label_dim: int = 50,
                 dropout: float = 0,
                 use_greedy_infer: bool = False,
                 **kwargs):
        super().__init__()
        if transform_dim > 0:
            word_embedding = build_word_embedding(**word_embedding)
            self.word_embedding = nn.Sequential(
                word_embedding,
                nn.Linear(word_embedding.output_dim, transform_dim)
            )
            feat_dim: int = transform_dim
        else:
            self.word_embedding = build_word_embedding(**word_embedding)
            feat_dim: int = self.word_embedding.output_dim
        if other_embedding is not None:
            self.other_embedding = DeepEmbedding(num_upos, **other_embedding)
            feat_dim += self.other_embedding.output_dim
        else:
            self.other_embedding = None

        if encoder is not None:
            self.encoder = build_encoder(feat_dim, dropout=dropout, **encoder)
            feat_dim = self.encoder.output_dim
        else:
            self.encoder = None

        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(feat_dim, (arc_dim + label_dim) * 2),
                nn.ReLU(inplace=True),
                TimestepDropout(p=dropout)
            )
        else:
            if encoder is None:
                raise ValueError("Encoder and MLP can't be None at same time!")
            self.mlp = None
            if feat_dim != 2 * (arc_dim + label_dim):
                raise ValueError("Wrong arc and label size!")

        self.arc_biaffine = ArcBiaffine(arc_dim)
        self.label_bilinear = LabelBilinear(label_dim, label_dim, num_label)
        # for calculate metrics precisely
        self.metrics_counter = OrderedDict({'arc': 0, 'label': 0, 'sample': 0})
        self.split_sizes = [arc_dim, arc_dim, label_dim, label_dim]
        self.use_greedy_infer = use_greedy_infer

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
        if self.other_embedding:
            upos = self.other_embedding(upos, **kwargs)
            feat = torch.cat([feat, upos], dim=2)
        if self.encoder is not None:
            feat = self.encoder(feat, **kwargs)  # unpack会去掉[SEP]那一列
            if feat.shape[1] == words.shape[1] - 1:
                mask, heads, deprel = remove_sep([mask, heads, deprel])
        if self.mlp is not None:
            feat = self.mlp(feat)
        feat = list(feat.split(self.split_sizes, dim=2))

        arc_pred = self.arc_biaffine(feat[0], feat[1])  # [N, L, L]

        # use gold or predicted arc to predict label
        batch_range = torch.arange(
            heads.shape[0], dtype=torch.long, device=heads.device).unsqueeze(1)
        if self.training:
            if heads is None:  # 一般不可能
                if self.use_greedy_infer:
                    heads = self.greedy_decoder(arc_pred, mask)
                else:
                    heads = self.mst_decoder(arc_pred, mask)
            head_pred = None
            feat[2] = feat[2][batch_range, heads].contiguous()
        else:
            if self.use_greedy_infer:
                head_pred = self.greedy_decoder(arc_pred, mask)
            else:
                head_pred = self.mst_decoder(arc_pred, mask)
            feat[2] = feat[2][batch_range, head_pred].contiguous()  # 用预测的head评测

        label_pred = self.label_bilinear(feat[2], feat[3])  # (B,L,C)
        output = {'arc': arc_pred, 'label': label_pred, 'head': head_pred}

        if self.training or self.evaluating:
            deprel = deprel.gather(1, heads)  # 按照头的顺序调整
            loss = self.loss(arc_pred, label_pred, heads, deprel, mask)
            output['loss'] = loss
        if self.evaluating:
            with torch.no_grad():
                output['metric'] = self.get_metrics(
                    head_pred, label_pred, heads, deprel, mask)

        return output

    @overrides
    def get_metrics(self,  # pylint:disable=arguments-differ
                    head_pred: torch.Tensor = None,
                    label_pred: torch.Tensor = None,
                    head_gt: torch.Tensor = None,
                    label_gt: torch.Tensor = None,
                    mask: torch.Tensor = None,
                    reset: bool = False) -> Dict[str, float]:
        """
        Evaluate the performance of prediction.
        reset = False， 计数，返回单次结果， True 用计数计算并清空
        """
        if reset:
            arc, label, sample = self.metrics_counter.values()
            for k in self.metrics_counter:
                self.metrics_counter[k] = 0
            return {'UAS': arc * 1.0 / sample, 'LAS': label * 1.0 / sample}

        if len(label_pred.shape) > len(label_gt.shape):
            pred_dim, indices_dim = 2, 1
            label_pred = label_pred.max(pred_dim)[indices_dim]
        mask = mask.long()
        # mask out <root> tag
        mask[:, 0] = 0
        head_pred_correct = (head_pred == head_gt).long() * mask
        label_pred_correct = (
            label_pred == label_gt).long() * head_pred_correct
        arc = head_pred_correct.sum().item()
        label = label_pred_correct.sum().item()
        sample = mask.sum().item()
        self.metrics_counter['arc'] += arc
        self.metrics_counter['label'] += label
        self.metrics_counter['sample'] += sample

        return {'UAS': arc * 1.0 / sample, 'LAS': label * 1.0 / sample}

    @staticmethod
    def loss(arc_pred: torch.Tensor,
             label_pred: torch.Tensor,
             arc_gt: torch.Tensor,
             label_gt: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
        """
        计算parser的loss
        :param arc_pred: [batch_size, seq_len, seq_len] 边预测logits
        :param label_pred: [batch_size, seq_len, num_label] label预测logits
        :param arc_gt: [batch_size, seq_len] 真实边的标注
        :param label_gt: [batch_size, seq_len] 真实类别的标注
        :return loss: scalar
        """

        batch_size, length, _ = arc_pred.shape
        flip_mask = mask.eq(False)
        _arc_pred = arc_pred.clone()
        _arc_pred = _arc_pred.masked_fill(
            flip_mask.unsqueeze(1), -float('inf'))
        arc_logits = F.log_softmax(_arc_pred, dim=2)
        label_logits = F.log_softmax(label_pred, dim=2)
        batch_index = torch.arange(
            batch_size, device=arc_logits.device, dtype=torch.long).unsqueeze(1)
        child_index = torch.arange(
            length, device=arc_logits.device, dtype=torch.long).unsqueeze(0)
        arc_loss = arc_logits[batch_index, child_index, arc_gt]
        label_loss = label_logits[batch_index, child_index, label_gt]

        arc_loss = arc_loss.masked_fill(flip_mask, 0)
        label_loss = label_loss.masked_fill(flip_mask, 0)
        arc_nll = -arc_loss.mean()
        label_nll = -label_loss.mean()
        return arc_nll + label_nll

    @staticmethod
    def is_best(metric: Dict[str, float], former: Dict[str, float]) -> bool:
        if metric['UAS'] > former['UAS']:
            return True
        elif metric['UAS'] == former['UAS']:
            return metric['LAS'] > former['LAS']
        else:
            return False
