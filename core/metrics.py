"""
"""

from argparse import Namespace
from collections import OrderedDict

import torch


def namespace_add(a, b):
    return Namespace(**{k: a.__dict__[k] + b.__dict__[k] for k in a.__dict__})


class Metric(object):
    """
    A very general abstract class representing a metric which can be accumulated.
    """

    def __init__(self):
        self.counter = self.counter_factory()
        self.best = self.metric_factory()

    def is_best(self, metric: OrderedDict[str, float]) -> bool:
        """
        根据key的顺序比较metric，在前者优先，默认数值越大越好。
        """
        for k, v in metric.items():
            if v > self.best[k]:
                self.best = metric
                return True
        return False

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor,
                 mask: torch.LongTensor) -> OrderedDict[str, float]:
        raise NotImplementedError

    def get_metric(self, counter=None, reset=False) -> OrderedDict[str, float]:
        raise NotImplementedError

    @staticmethod
    def counter_factory(**kwargs) -> Namespace:
        raise NotImplementedError

    @staticmethod
    def metric_factory(**kwargs) -> OrderedDict:
        raise NotImplementedError


class TaggingMetric(Metric):
    def __init__(self, ignore_index: int = 0):
        super().__init__()
        self.ignore_index = ignore_index

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor,
                 mask: torch.LongTensor) -> OrderedDict[str, float]:
        batch = self.counter_factory()

        mask = (gold_labels != self.ignore_index).long() * mask  # 只看标注
        batch.total = mask.sum().item()
        batch.positive = ((predictions != self.ignore_index).long() * mask).sum().item()
        batch.correct = ((predictions == gold_labels).long() * mask).sum().item()

        self.counter = namespace_add(self.counter, batch)

        return self.get_metric(batch)

    @staticmethod
    def counter_factory(total=0, positive=0, correct=0) -> Namespace:
        return Namespace(total=total, positive=positive, correct=correct)

    @staticmethod
    def metric_factory(f1=0, recall=0, precision=0) -> OrderedDict:
        return OrderedDict(F1=f1, recall=recall, precision=precision)

    def get_metric(self, counter=None, reset=False):
        counter = counter or self.counter

        recall = counter.positive / counter.total
        precision = counter.correct / counter.total
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        if reset:
            self.counter = self.counter_factory()

        return self.metric_factory(f1, recall, precision)
