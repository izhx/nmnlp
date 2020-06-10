"""
Dataset Abstract class
"""

from typing import Iterable, Dict, Tuple, List, Set, Any, Union
import itertools

from torch.utils.data import Dataset, ConcatDataset

from ..common.constant import KEY_TRAIN, PRETRAIN_POSTFIX
from .vocabulary import Vocabulary


class DataSet(Dataset):
    """
    data container.
    """
    index_fields: Set[str]  # 需要index的field

    def __init__(self, data: List, tokenizer: Any = None,
                 pretrained_fields: Set[str] = (), in_memory: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.pretrained_fields = {k + PRETRAIN_POSTFIX for k in pretrained_fields}
        self.in_memory = in_memory  # 完成数据不在内存的代码
        self.indexed = False

    def __getitem__(self, item):
        # if self.in_memory:
        return self.data[item]
        # else:
        #     raise NotImplementedError('Feature in coming.')

    def __add__(self, other) -> ConcatDataset:
        return ConcatDataset([self, other])

    def __iter__(self) -> Iterable:
        return iter(self.data)

    def __len__(self):
        # if self.in_memory:
        return len(self.data)
        # else:
        #     raise NotImplementedError('Feature in coming.')

    @classmethod
    def build(cls, path, kind: str = KEY_TRAIN) -> Union['DataSet', List, Dict]:
        raise NotImplementedError

    def text_to_instance(self, *inputs) -> Any:
        raise NotImplementedError

    def collate_fn(self, batch) -> Tuple[Dict, Dict]:
        raise NotImplementedError

    def index_dataset(self, vocab: Vocabulary):
        """
        Data should be indexed or other operation by this method.
        """
        if not self.indexed:
            self.data = [self.instance_to_index(i, vocab) for i in self.data]
            self.indexed = True

    def instance_to_index(self, instance, vocab: Vocabulary) -> Any:
        for key in itertools.chain(self.index_fields, self.pretrained_fields):
            instance[key] = vocab.tokens_to_indices(instance[key], key)
        return instance
