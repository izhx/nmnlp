from typing import Any, List, Dict, Tuple
import os
import logging
from collections import OrderedDict, defaultdict
from overrides import overrides

import torch
from conllu import parse_incr

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

        word_pieces = dict()
        if self.tokenizer is not None:
            tokens = ['<root>']
            for i, word in enumerate(words[1:], 1):
                _tokens = self.tokenizer.tokenize(word)
                tokens.append(_tokens[0])
                if len(_tokens) > 1:
                    word_pieces[i] = [self.tokenizer.vocab[p] for p in _tokens]
        else:
            tokens = [word.lower() for word in words]

        fields['word_ids'] = ids
        fields["words"] = tokens
        fields['lemma'] = lemma
        fields["upos"] = upos_tags
        fields['word_pieces'] = word_pieces
        if dependencies is not None:
            fields["deprel"], fields["heads"] = dependencies

        fields["metadata"] = {"words": words, "pos": upos_tags,
                              "lang": self.lang, 'len': len(ids)}
        return fields

    def collate_fn(self, batch) -> Dict[str, Any]:
        used_keys = ('words', 'upos', 'deprel', 'heads', 'word_ids')

        ids_sorted = sorted(range(len(batch)),
                            key=lambda x: batch[x]['metadata']['len'],
                            reverse=True)

        max_len = batch[ids_sorted[0]]['metadata']['len'] + 1  # for bert
        result = defaultdict(lambda: torch.zeros(
            len(batch), max_len, dtype=torch.long))
        result['mask'] = torch.zeros((len(batch), max_len)).bool()
        result['seq_lens'], result['sentences'] = list(), list()
        result['word_pieces'] = dict()

        for i, o in zip(range(len(batch)), ids_sorted):
            seq_len = len(batch[o]['words'])
            result['seq_lens'].append(seq_len)
            result['sentences'].append(batch[o]['metadata']['words'])
            result['mask'][i, 1:seq_len] = True
            for key in used_keys:
                result[key][i, :seq_len] = torch.LongTensor(batch[o][key])
            for w, piece in batch[o]['word_pieces'].items():
                result['word_pieces'][(i, w)] = torch.LongTensor(piece)

        return result
