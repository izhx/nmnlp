from typing import List, Iterator, Dict, Tuple, Any

from tjunlp.data import DatasetReader


class Predictor(object):
    """
        A ``Predictor`` is a thin wrapper around an model.
    """

    def __init__(self, model, dataset_reader: DatasetReader) -> None:
        self._model = model
        self._dataset_reader = dataset_reader
