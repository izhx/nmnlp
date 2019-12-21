from typing import Any, List, Dict, Tuple
import os
import logging
from collections import OrderedDict
from overrides import overrides

import torch
from conllu import parse_incr

from tjunlp.core.vocabulary import DEFAULT_PADDING_INDEX
from tjunlp.core.dataset import DataSet

logger = logging.getLogger(__name__)

_ROOT = OrderedDict([('id', 0), ('form', '<root>'), ('lemma', ''),
                     ('upostag', 'root'), ('xpostag', None), ('feats', None),
                     ('head', 0), ('deprel', 'root'), ('deps', None),
                     ('misc', None)])


class ConlluDataset(DataSet):
    index_fields = ('words', 'lemma', 'upos', 'deprel')

    def __init__(self,
                 file_path: str,
                 tokenizer=None,
                 lang: str = 'en',
                 multi_lang: bool = False,
                 use_language_specific_pos: bool = False,
                 **kwargs):
        self.lang = lang
        self.multi_lang = multi_lang
        self.use_language_specific_pos = use_language_specific_pos
        super().__init__(file_path, tokenizer)  # 不可调换顺序

    def read(self, file_path: str) -> List:
        data = list()
        with open(file_path, "r") as conllu_file:
            logger.info(
                "Reading UD instances from conllu dataset at: %s", file_path)

            for annotation in parse_incr(conllu_file):
                # if len(annotation) < 3:
                #     print(annotation)
                #     continue
                annotation = [
                    x for x in annotation if isinstance(x["id"], int)]
                if annotation[0]['id'] == 0:
                    for i in range(len(annotation)):
                        annotation[i]['id'] += 1
                annotation.insert(0, _ROOT)
                ids = [x["id"] for x in annotation]
                heads = [x["head"] for x in annotation]
                lemma = [x["lemma"] for x in annotation]
                deprel = [x["deprel"] for x in annotation]
                if self.multi_lang:
                    words = [f'{x["form"]}_{self.lang}' for x in annotation]
                else:
                    words = [x["form"] for x in annotation]
                # if self.use_language_specific_pos:
                #     pos_tags = [x["xpostag"] for x in annotation]
                upos_tag = [x["upostag"] for x in annotation]
                data.append(self.text_to_instance(
                    ids, words, lemma, upos_tag, (deprel, heads)))
        return data

    @overrides
    def text_to_instance(self, ids: List[int], words: List[str], lemma: List[str],
                         upos_tags: List[str], dependencies: Tuple = None):
        fields: Dict[str, object] = {}

        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(" ".join(words))
        else:
            tokens = [t.lower() for t in words]

        fields['word_ids'] = ids
        fields["words"] = tokens
        fields['lemma'] = lemma
        fields["upos"] = upos_tags
        if dependencies is not None:
            fields["deprel"], fields["heads"] = dependencies

        fields["metadata"] = {"words": words, "pos": upos_tags,
                              "lang": self.lang, 'len': len(ids)}
        return fields

    def collate_fn(self, batch) -> Dict[str, Any]:
        result = dict()  # batch, seq
        used_keys = ('words', 'upos', 'deprel', 'heads')

        ids_sorted = sorted(range(len(batch)),
                            key=lambda x: batch[x]['metadata']['len'],
                            reverse=True)

        max_len = batch[ids_sorted[0]]['metadata']['len']

        for i, o in zip(range(len(batch)), ids_sorted):
            seq_len = len(batch[o]['words'])
            result.setdefault('seq_lens', list()).append(seq_len)
            for key in used_keys:
                result.setdefault(key, torch.zeros((len(
                    batch), max_len), dtype=torch.int64))[i, :seq_len] = torch.LongTensor(
                        batch[o][key])

            heads = torch.LongTensor(batch[o]['heads'])
            if torch.any(heads < 0) or torch.any(heads >= seq_len):
                raise Exception("?????????")

        result['mask'] = torch.eq(
            result['words'], DEFAULT_PADDING_INDEX).eq(False)

        return result
