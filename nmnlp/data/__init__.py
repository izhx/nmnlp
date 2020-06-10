from typing import Dict, Any

from tjunlp.common.constant import KEY_TRAIN, KEY_DEV, KEY_TEST
from tjunlp.core.dataset import DataSet
from tjunlp.data.conll import ConlluDataset

_DATASET = {
    'Conllu': ConlluDataset
}


def build_dataset(name: str, data_dir: str, read_test: bool = False,
                  **kwargs) -> Dict[str, Any]:
    dataset = dict()
    dataset[KEY_TRAIN] = _DATASET[name].build(data_dir, KEY_TRAIN, **kwargs)
    dataset[KEY_DEV] = _DATASET[name].build(data_dir, KEY_DEV, **kwargs)
    if read_test:
        dataset[KEY_TEST] = _DATASET[name].build(
            data_dir, KEY_TEST, **kwargs)

    return dataset


def index_dataset(dataset, vocab):
    if isinstance(dataset, DataSet):
        dataset.index_dataset(vocab)
    elif isinstance(dataset, dict):
        for key in dataset:
            index_dataset(dataset[key], vocab)
    elif isinstance(dataset, list):
        for i in range(len(dataset)):
            index_dataset(dataset[i], vocab)
