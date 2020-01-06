import os
import logging
from typing import Iterable, Dict, Tuple, List, Any

from torch.utils.data import Dataset, ConcatDataset

from tjunlp.common.checks import ConfigurationError
from tjunlp.core.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

KIND_TRAIN, KIND_DEV, KIND_TEST = 'train', 'dev', 'test'


class DataSet(Dataset):
    """
    data container.
    """
    index_fields: Tuple[str]  # 需要index的field

    def __init__(self, path: str, kind: str = KIND_TRAIN, tokenizer=None,
                 in_memory: bool = True):
        self.tokenizer = tokenizer
        self.in_memory = in_memory  # TODO(izhx): 完成数据不在内存的代码
        self.indexed = False

        if os.path.exists(path):
            if in_memory:
                self.data = self.read(path, kind)
                if not self.data:
                    raise ConfigurationError(f"No data at: {path}")
        else:
            raise ConfigurationError(f"File not exist! Please check : {path}")

    def __getitem__(self, item):
        if self.in_memory:
            return self.data[item]
        else:
            raise NotImplementedError('Feature in coming.')

    def __add__(self, other):
        return ConcatDataset([self, other])

    # def __iter__(self) -> Iterable:
    #     return iter(self.data)

    def __len__(self):
        if self.in_memory:
            return len(self.data)
        else:
            raise NotImplementedError('Feature in coming.')

    def read(self, path, kind) -> List:
        raise NotImplementedError

    def text_to_instance(self, *inputs):
        raise NotImplementedError

    def collate_fn(self, batch) -> Tuple[Dict, Dict]:
        raise NotImplementedError

    def index_dataset(self, vocab: Vocabulary):
        """
        Data should be indexed or other operation by this method.
        """
        self.data = [self.instance_to_index(ins, vocab) for ins in self.data]
        self.indexed = True

    @classmethod
    def instance_to_index(cls, instance, vocab: Vocabulary):
        for key in cls.index_fields:
            instance[key] = vocab.tokens_to_indices(instance[key], key)
        return instance
