from typing import Dict

from tjunlp.core.dataset import DataSet, KIND_TRAIN, KIND_DEV, KIND_TEST
from tjunlp.data.conll import ConlluDataset

_DATASET = {
    'Conllu': ConlluDataset
}


DATA_FILE_KEY_POSTFIX = '_file'


def build_dataset(name: str, data_dir: str, read_test: bool = False,
                  **kwargs) -> Dict[str, DataSet]:
    dataset = dict()
    dataset[KIND_TRAIN] = _DATASET[name](data_dir, KIND_TRAIN, **kwargs)
    dataset[KIND_DEV] = _DATASET[name](data_dir, KIND_DEV, **kwargs)
    if read_test:
        dataset[KIND_TEST] = _DATASET[name](data_dir, KIND_TEST, **kwargs)

    return dataset
