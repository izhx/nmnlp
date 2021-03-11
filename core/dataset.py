"""
Dataset Abstract class
"""

from typing import Dict, List, Set, Any, Union, Iterable
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from ..common.constant import PRETRAIN_POSTFIX
from .vocabulary import Vocabulary


class DataSet(Dataset):
    """
    data container.
    """
    index_fields: Set[str] = None  # 需要index的field，也就是建立词表的

    def __init__(self, data: Iterable = None, pretrained_fields: Set[str] = ()):
        self.data = data or list()
        self.pretrained_fields = pretrained_fields
        self.indexed = False
        self.vec_fields = list()
        self.int_fields = list()

    def __getitem__(self, idx) -> Dict:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def build(cls, path, kind) -> Union['DataSet', List, Dict]:
        raise NotImplementedError

    # 为了简单，就不太优雅
    def collate_fn(self, batch) -> Dict[str, Any]:
        lengths = [len(ins[self.vec_fields[0]]) for ins in batch]
        vec_dict = defaultdict(lambda: torch.zeros(len(batch), max(lengths), dtype=torch.long))
        int_dict = defaultdict(lambda: torch.zeros(len(batch), dtype=torch.long))

        for i, (ins, seq_len) in enumerate(zip(batch, lengths)):
            vec_dict['mask'][i, :seq_len] = 1
            for field in self.int_fields:
                int_dict[field][i] = ins[field]
            for field in self.vec_fields:
                # torch.nn.functional.pad(tensor,(0, max_len - seq_len))
                vec_dict[field][i, :seq_len] = torch.LongTensor(ins[field])
        else:
            int_dict['lengths'] = torch.LongTensor(lengths)
            vec_dict.update(int_dict)

        return vec_dict, batch

    def index_with(self, vocab: Vocabulary) -> None:
        """
        Data should be indexed or other operation by this method.
        """
        def index_field(field, namespace):
            if isinstance(field, list):
                return vocab.indices_of(field, namespace)
            elif isinstance(field, str):
                return vocab.index_of(field, namespace)
            raise RuntimeError("不支持的操作!")

        field_pairs = ((f, f + PRETRAIN_POSTFIX) for f in self.pretrained_fields)

        if not self.indexed:
            for i, ins in enumerate(self.data):
                for field, pretrain in field_pairs:
                    ins[pretrain] = index_field(ins[field], pretrain)
                for field in self.index_fields:
                    ins[field] = index_field(ins[field], field)
            self.indexed = True

        self.vec_fields = [k for k, v in self.data[0].items() if is_vec(v)]
        self.int_fields = [k for k, v in self.data[0].items() if isinstance(v, int)]


def is_vec(obj):
    if isinstance(obj, list):
        for i in obj:
            if isinstance(i, list) and is_vec(i):
                continue
            if not isinstance(i, int):
                break
        else:
            return True
    return False
