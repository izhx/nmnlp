"""
Conllu dataset.
"""

from typing import Any, List, Dict
import os
import glob
import random
import logging
from collections import OrderedDict, defaultdict
from overrides import overrides

import torch
from conllu import parse_incr

from tjunlp.common.tqdm import Tqdm
from tjunlp.core.dataset import DataSet, KIND_TRAIN

logger = logging.getLogger(__name__)

_ROOT = OrderedDict([('id', 0), ('form', '<root>'), ('lemma', ''),
                     ('upostag', 'root'), ('xpostag', None), ('feats', None),
                     ('head', 0), ('deprel', 'root'), ('deps', None),
                     ('misc', None)])


class ConlluDataset(DataSet):
    ud_keys = ('id', 'form', 'upostag', 'head', 'deprel')  # 暂时不用 'lemma'
    index_fields = ('words', 'upostag', 'deprel')
    max_len = 128

    def __init__(self,
                 data: str,
                 tokenizer: Any = None,
                 langs: list = None,
                 min_len: int = 2):
        super().__init__(data, tokenizer)
        self.langs = langs
        self.min_len = min_len
        # self.use_language_specific_pos = use_language_specific_pos
        self.source2id = dict()
        self.percentage = defaultdict(int)
        self.counter = defaultdict(int)  # int() = 0
        self.droped = defaultdict(int)

    @classmethod
    def build(cls,
              path: str,
              kind: str = KIND_TRAIN,
              tokenizer: Any = None,
              langs: list = None,
              min_len: int = 2):
        path = os.path.normpath(path)
        if not os.path.isdir(path):
            raise ValueError(f'"{path}" is not a dir!')
        path_list = list()
        for lang in langs:
            path = f"{path}/*/{lang}_*ud-{kind}.conllu"
            path_list.extend(glob.glob(path))

        path_list = [os.path.normpath(p) for p in path_list]

        if kind == KIND_TRAIN:
            dataset = cls([], tokenizer, langs, min_len)
            for path in Tqdm(path_list):
                dataset.read_one(path)
            return dataset.stat(len(path_list) > 1)
        dataset = defaultdict(lambda: cls([], tokenizer, None, min_len))
        for path in Tqdm(path_list):
            lang = path.split('/')[-1].split('_')[0]
            dataset[lang].read_one(path)
            dataset[lang].langs = [lang]
        return dict(dataset)

    def read_one(self, file_path: str) -> List:
        lang = file_path.split('/')[-1].split('_')[0]
        if lang not in self.source2id:
            source_id = len(self.source2id)
            self.source2id[lang] = source_id
        else:
            source_id = self.source2id[lang]
        total_num, droped_num = self._read(file_path, source_id)
        name = '/'.join(file_path.split('/')[-2:])
        self.counter[name], self.droped[name] = total_num, droped_num
        self.percentage[source_id] += total_num - droped_num
        Tqdm.write(
            f"===> [{name}]  totally {total_num}, droped {droped_num}.")
        return self

    def _read(self, file_path: str, source_id: int = 0):
        total_num, droped_num, a, b = 0, 0, 0, 0
        with open(file_path, mode="r", encoding="UTF-8") as conllu_file:
            for annotation in parse_incr(conllu_file):
                # print(annotation)
                annotation = [
                    x for x in annotation if isinstance(x["id"], int)]
                if random.random() < 0.1:
                    for x in annotation:
                        a += 1
                        if x['form'] == '_':
                            b += 1

                if annotation[0]['id'] == 0:
                    for i in range(len(annotation)):
                        annotation[i]['id'] += 1
                annotation.insert(0, _ROOT)
                total_num += 1
                if self.max_len > len(annotation) > self.min_len:
                    self.data.append(
                        self.text_to_instance(annotation, source_id))
                else:
                    # Tqdm.write(annotation[1]['form'])
                    droped_num += 1
            if b/a > 0.2:
                Tqdm.write(f"=========> {file_path} ????.'")
        return total_num, droped_num

    def stat(self, log: bool = False):
        t, d = 0, 0
        for k in self.droped.keys():
            t += self.counter[k]
            d += self.droped[k]
            self.counter[k] -= self.droped[k]
        if log:
            print(f'===> Totally {t}, droped {d} one word instence.')
        t -= d
        self.percentage = {
            k: float(v)/t for k, v in self.percentage.items()}
        return self

    @overrides
    def text_to_instance(self, annotation: List, source: str):
        fields = defaultdict(list)
        for x in annotation:
            for k in self.ud_keys:
                fields[k].append(x[k])

        words, pieces = fields['form'], dict()
        if self.tokenizer is not None:
            tokens = ['<root>']
            for i, word in enumerate(words[1:], 1):
                if word == '_' and annotation[i]['lemma'] != '_':
                    word = annotation[i]['lemma']
                piece = self.tokenizer.tokenize(word)
                if len(piece) > 0:
                    tokens.append(piece[0])
                    if len(piece) > 1:
                        pieces[i] = [self.tokenizer.vocab[p] for p in piece]
                else:
                    tokens.append(word)
        else:
            tokens = [word.lower() for word in words]

        for i, h in enumerate(fields['head']):
            if h is None:
                fields['head'][i] = 0  # 指向虚根，在UD_Portuguese-Bosque等会有None

        fields["words"] = tokens
        fields["word_pieces"] = pieces
        fields["metadata"] = {'len': len(annotation), 'source': source}
        return dict(fields)

    def collate_fn(self, batch) -> Dict[str, Any]:
        ids_sorted = sorted(
            range(len(batch)), key=lambda i: batch[i]['metadata']['len'], reverse=True)

        max_len = batch[ids_sorted[0]]['metadata']['len'] + 1  # for bert
        result = defaultdict(lambda: torch.zeros(
            len(batch), max_len, dtype=torch.long))
        result['mask'] = torch.zeros((len(batch), max_len)).bool()
        result['seq_lens'], result['sentences'] = list(), list()
        result['word_pieces'] = dict()

        for i, origin in zip(range(len(batch)), ids_sorted):
            seq_len = len(batch[origin]['words'])
            result['seq_lens'].append(seq_len)
            result['sentences'].append(batch[origin]['form'])
            result['mask'][i, 1:seq_len] = True
            for key in ('words', 'upostag', 'deprel', 'head', 'id'):
                result[key][i, :seq_len] = torch.LongTensor(batch[origin][key])
            for w, piece in batch[origin]['word_pieces'].items():
                result['word_pieces'][(i, w)] = torch.LongTensor(piece)

        return result
