import logging
from typing import Iterable, Dict, Tuple, List

from tjunlp.common.checks import ConfigurationError
from tjunlp.core.vocabulary import Vocabulary
from tjunlp.core.instance import Instance

logger = logging.getLogger(__name__)


class DataSet(object):
    """
    data container.
    """

    def __init__(self, data: Iterable[Instance], index_func, in_memory: bool = True):
        """

        :param reader:
        :param in_memory: data is stored in memory or cache dir.
        """
        self.in_memory = in_memory  # TODO(izhx): 完成数据不在内存的代码
        self.data = data
        self.index_func = index_func
        self.indexed = False

        if in_memory:
            self.data = [ins for ins in self.data]
            if not self.data:
                raise ConfigurationError(
                    "No instances were read from the given filepath. "
                    "Is the path correct?"
                )

    def __getitem__(self, item):
        if self.in_memory:
            return self.data[item]
        else:
            raise NotImplementedError('Feature in coming.')

    def __iter__(self) -> Iterable[Instance]:
        return iter(self.data)

    def __len__(self):
        if self.in_memory:
            return len(self.data)
        else:
            raise NotImplementedError('Feature in coming.')

    def __add__(self, other):
        # TODO(izhx): def add operation
        pass

    def index_dataset(self, vocab: Vocabulary, **kwargs):
        """
        Data should be indexed or other operation by this method.
        """
        self.data = [self.index_func(ins, vocab, **kwargs) for ins in self.data]
        self.indexed = True
