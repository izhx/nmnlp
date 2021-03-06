"""
"""

from typing import Dict
from argparse import Namespace

import torch


def a_better_than_b(a, b):
    for k, v in a.items():
        if v > b[k]:
            return True
        elif v < b[k]:
            return False
    return False


def namespace_add(a, b):
    return Namespace(**{k: a.__dict__[k] + b.__dict__[k] for k in a.__dict__})


class Metric(object):
    """
    A very general abstract class representing a metric which can be accumulated.
    """

    def __init__(self):
        self.counter = self.counter_factory()
        self.best = None

    def is_best(self, metric: Dict) -> bool:
        """
        根据key的顺序比较metric，在前者优先，默认数值越大越好。
        """
        if self.best is None or a_better_than_b(metric, self.best):
            self.best = metric
            return True
        return False

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor,
                 mask: torch.LongTensor) -> Dict:
        """
        每个batch调用，更新counter，计算当前batch的分数并返回。
        """
        raise NotImplementedError

    def get_metric(self, counter=None, reset=False) -> Dict:
        """
        用counter计算出metric。
        """
        raise NotImplementedError

    @staticmethod
    def counter_factory(**kwargs) -> Namespace:
        raise NotImplementedError

    @staticmethod
    def metric_factory(**kwargs) -> Dict:
        """
        注意按重要性排列参数。
        """
        raise NotImplementedError


class TaggingMetric(Metric):
    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor,
                 mask: torch.LongTensor) -> Dict:
        batch = self.counter_factory()

        mask = (gold_labels != self.ignore_index).long() * mask  # 只看标注
        batch.total = mask.sum().item()
        batch.positive = ((predictions != self.ignore_index).long() * mask).sum().item()
        batch.correct = ((predictions == gold_labels).long() * mask).sum().item()

        self.counter = namespace_add(self.counter, batch)

        return self.get_metric(batch)

    @staticmethod
    def counter_factory(total=0, positive=0, correct=.0) -> Namespace:
        return Namespace(total=total, positive=positive, correct=correct)

    @staticmethod
    def metric_factory(f1=.0, recall=.0, precision=.0) -> Dict:
        return dict(F1=f1, recall=recall, precision=precision)

    def get_metric(self, counter=None, reset=False) -> Dict:
        c = counter or self.counter
        total, correct, positive = c.total, c.correct, c.positive
        recall = 0 if total == 0 else correct / total
        precision = 0 if positive == 0 else correct / positive

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        if reset:
            self.counter = self.counter_factory()

        return self.metric_factory(f1, recall, precision)
