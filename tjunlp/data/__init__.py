from typing import Dict

from tjunlp.core.dataset import DataSet
from tjunlp.data.conll import ConlluDataset

_DATASET = {
    'Conllu': ConlluDataset
}


def build_dataset(name: str, data_dir: str, train_file: str,
                  dev_file: str = None, read_other: bool = False,
                  **kwargs) -> Dict[str, DataSet]:
    dataset = dict()
    if train_file:
        dataset['train'] = _DATASET[name](f"{data_dir}/{train_file}", **kwargs)
    if dev_file:
        dataset['dev'] = _DATASET[name](f"{data_dir}/{dev_file}", **kwargs)

    if not read_other:
        return dataset

    for key in kwargs:
        if not key.endswith('_file'):
            continue
        dataset[key[:-5]] = _DATASET[name](f"{data_dir}/{kwargs[key]}", **kwargs)

    return dataset
