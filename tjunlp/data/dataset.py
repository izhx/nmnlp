from typing import Any, List, Dict, Tuple
import os
import logging
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, ConcatDataset
from conllu import parse_incr

from tjunlp.common.checks import ConfigurationError
from tjunlp.core.vocabulary import Vocabulary, DEFAULT_PADDING_INDEX

logger = logging.getLogger(__name__)

_ROOT = OrderedDict([('id', 0), ('form', '<root>'), ('lemma', ''),
                     ('upostag', 'root'), ('xpostag', None), ('feats', None),
                     ('head', 0), ('deprel', 'root'), ('deps', None),
                     ('misc', None)])


class ConlluDataset(Dataset):
    def __init__(self, path: str, tokenizer=None, lang: str = 'en',
                 multi_lang: bool = False,
                 use_language_specific_pos: bool = False,
                 **kwargs):
        self.target_fields = ['rels', 'heads']
        self.tokenizer = tokenizer
        self.lang = lang
        self.multi_lang = multi_lang
        self.use_language_specific_pos = use_language_specific_pos
        self.indexed = False
        self.vocab = None

        if os.path.exists(path):
            self.data = self.read(path)
            if not self.data:
                raise ConfigurationError(f"No data at: {path}")
        else:
            raise ConfigurationError(f"File not exist! Please check the path: {path}")

    def __getitem__(self, item):
        return self.data[item]

    def __add__(self, other):
        return ConcatDataset([self, other])

    def __len__(self):
        return len(self.data)

    def read(self, file_path) -> List:
        data = list()
        with open(file_path, "r") as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            for annotation in parse_incr(conllu_file):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by None here as elided words have a non-integer word id,
                # and are replaced with None by the conllu python library.

                annotation = [x for x in annotation if isinstance(x["id"], int)]
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
                data.append(self.text_to_instance(ids, words, lemma, upos_tag, (deprel, heads)))
        return data

    def text_to_instance(
            self,
            ids: List[int],
            words: List[str],
            lemma: List[str],
            upos_tags: List[str],
            dependencies: Tuple = None,
    ):  # TODO(izhx) 加一个虚根？
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
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["deprel"], fields["heads"] = dependencies

        fields["metadata"] = {"words": words, "pos": upos_tags,
                              "lang": self.lang, 'sent_len': len(ids)}
        return fields

    @staticmethod
    def instance_to_index(instance, vocab: Vocabulary):
        for key in ('words', 'lemma', 'upos', 'deprel'):
            instance[key] = vocab.tokens_to_indices(instance[key], key)
        return instance

    def index_dataset(self, vocab: Vocabulary):
        """
        Data should be indexed or other operation by this method.
        """
        self.data = [self.instance_to_index(ins, vocab) for ins in self.data]
        self.indexed = True
        self.vocab = vocab

    def collate_fn(self, batch) -> Dict[str, Any]:
        result = dict()  # batch, seq

        ids_sorted = sorted(range(len(batch)),
                            key=lambda x: batch[x]['metadata']['sent_len'],
                            reverse=True)

        max_len = batch[ids_sorted[0]]['metadata']['sent_len']

        for i, o in zip(range(len(batch)), ids_sorted):
            sent_len = len(batch[o]['words'])
            result.setdefault('sent_lens', list()).append(sent_len)
            for key in ('word_ids', 'words', 'lemma', 'upos', 'deprel', 'heads'):
                result.setdefault(key, torch.zeros((len(batch), max_len), dtype=torch.int64))[i,
                :sent_len] = torch.tensor(batch[o][key], dtype=torch.int64)

            heads = torch.tensor(batch[o]['heads'], dtype=torch.int64)
            if torch.any(heads < 0) or torch.any(heads >= sent_len):
                raise Exception("?????????")

            result.setdefault('pretrained', torch.zeros((len(batch), max_len), dtype=torch.int64))[
            i, :sent_len] = torch.tensor(batch[o]['words'], dtype=torch.int64)

        result['word_mask'] = torch.eq(result['words'], DEFAULT_PADDING_INDEX).bool()

        for key in ('word_ids', 'words', 'lemma', 'upos', 'deprel', 'heads'):
            if torch.any(result[key] < 0):
                raise Exception("?????????")

        return result
