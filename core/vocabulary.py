"""
A Vocabulary maps strings to integers, allowing for strings to be mapped to an
out-of-vocabulary token.
"""

import warnings

from typing import Dict, Iterable, List, Union, Optional
from argparse import Namespace
from itertools import chain
from collections import defaultdict

from ..common.constant import PRETRAIN_POSTFIX
from ..common.util import output


DEFAULT_PADDING_TOKEN = "<pad>"
DEFAULT_PADDING_INDEX = 0
DEFAULT_OOV_TOKEN = "<unk>"
DEFAULT_OOV_INDEX = 1


class Vocabulary(object):
    def __init__(self,
                 field_token_counts: Dict[str, List[str]],
                 closed_fields: Iterable[str],
                 min_count: Dict[str, int] = {},
                 pretrained_files: Dict[str, str] = {},
                 only_include_pretrained_words: bool = False,
                 padding_token: str = DEFAULT_PADDING_TOKEN,
                 oov_token: str = DEFAULT_OOV_TOKEN):
        """
        """
        min_count = min_count or dict()
        pretrained_files = pretrained_files or dict()
        self.padding_token = padding_token
        self.oov_token = oov_token
        self.counter = field_token_counts
        self.token_to_index = {k: {padding_token: 0} for k in field_token_counts}
        self.index_to_token = {k: {0: padding_token} for k in field_token_counts}

        for field, token_counts in field_token_counts.items():
            if field not in closed_fields:
                self.add_token_to_field(self.oov_token, field)

            filed_min_count = min_count.get(field, 1)

            if field in pretrained_files:
                pretrained_set = set(_read_pretrained_tokens(pretrained_files[field]))
                if only_include_pretrained_words:
                    for token, count in token_counts:
                        if token in pretrained_set and count >= filed_min_count:
                            self.add_token_to_field(token, field)
                else:  # 分成两个字典
                    field_pretrained = field + PRETRAIN_POSTFIX
                    token_counts = {k: v for k, v in token_counts.items() if v > filed_min_count}
                    for token in pretrained_set:
                        self.add_token_to_field(token, field)
                        self.add_token_to_field(token, field_pretrained)
                        if token in token_counts:
                            token_counts.pop(token)
                    for token, count in token_counts.items():
                        self.add_token_to_field(token, field)
            else:
                for token, count in token_counts.items():
                    if count >= filed_min_count:
                        self.add_token_to_field(token, field)

    @classmethod
    def from_data(cls,
                  datasets: Union[List, Namespace],
                  closed_fields: Iterable[str] = (),
                  min_count: Dict[str, int] = None,
                  pretrained_files: Optional[Dict[str, str]] = {},
                  only_include_pretrained_words: bool = False,
                  padding_token: str = DEFAULT_PADDING_TOKEN,
                  oov_token: str = DEFAULT_OOV_TOKEN) -> 'Vocabulary':
        """
        从train和dev数据集构建各field的词表。
        """
        field_token_counts = defaultdict(lambda: defaultdict(int))

        if isinstance(datasets, Namespace):
            create_fields = datasets.train.index_fields
            instances = chain(datasets.train, datasets.dev)
        elif isinstance(datasets, List):
            create_fields = datasets[0].index_fields
            instances = chain(*datasets)
        else:
            warnings.warn("dataset类型错误!")

        for instance in instances:
            for field in create_fields:
                for token in instance[field]:
                    field_token_counts[field][token] += 1

        return cls(dict(field_token_counts),
                   closed_fields,
                   min_count,
                   pretrained_files,
                   only_include_pretrained_words,
                   padding_token,
                   oov_token)

    def set_field(self, tokens: Iterable[str], field: str) -> None:
        """
        按列表顺序，建立新field，如果存在将被覆盖。
        """
        if field in self.token_to_index:
            warnings.warn(f"Field {field} 已存在，将被覆盖！")
        self.token_to_index[field] = {k: i for i, k in enumerate(tokens)}
        self.index_to_token[field] = {i: k for i, k in enumerate(tokens)}

    def add_token_to_field(self, token: str, field: str) -> int:
        """
        Adds ``token`` to the index, if it is not already present.  Either way, we return the index of
        the token.
        """
        if not isinstance(token, str):
            raise ValueError("Vocabulary tokens must be strings, or saving and loading will break."
                             "  Got %s (with type %s)" % (repr(token), type(token)))
        if token not in self.token_to_index[field]:
            index = len(self.token_to_index[field])
            self.token_to_index[field][token] = index
            self.index_to_token[field][index] = token
            return index
        else:
            return self.token_to_index[field][token]

    def add_tokens_to_field(self, tokens: Iterable[str], field: str) -> List[int]:
        """
        Adds ``tokens`` to the index, if they are not already present.  Either way, we return the
        indices of the tokens in the order that they were given.
        """
        return [self.add_token_to_field(token, field) for token in tokens]

    def index_of(self, token: str, field: str) -> int:
        if token in self.token_to_index[field]:
            return self.token_to_index[field][token]
        else:
            try:
                return self.token_to_index[field][self.oov_token]
            except KeyError:
                Warning(f'Field: {field}')
                Warning(f'Token: {token}')
                raise

    def indices_of(self, tokens: Iterable, field: str) -> List[int]:
        return [self.index_of(token, field) for token in tokens]

    def size_of(self, field: str) -> int:
        return len(self.index_to_token[field])

    def __repr__(self) -> str:
        base_string = "Vocabulary with fields: "
        fields = [f"{name}, Size: {self.size_of(name)} ||"
                  for name in self.index_to_token]
        return " ".join([base_string] + fields)


def _read_pretrained_tokens(embeddings_file: str) -> List[str]:
    # Moving this import to the top breaks everything (cycling import, I guess)
    from ..embedding.embedding import EmbeddingsTextFile

    output(f'Reading pretrained tokens from: <{embeddings_file}>')
    tokens: List[str] = []
    with EmbeddingsTextFile(embeddings_file) as file:
        for line_number, line in enumerate(file, start=1):
            token_end = line.find(' ')
            if token_end >= 0:
                token = line[:token_end]
                tokens.append(token)
            else:
                line_begin = line[:20] + '...' if len(line) > 20 else line
                warnings.warn('Skipping line number %d: %s',
                              line_number, line_begin)
    return tokens
