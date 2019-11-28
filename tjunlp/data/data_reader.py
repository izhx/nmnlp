from typing import Iterable, List, Tuple, Dict
import os
import logging
from overrides import overrides

from conllu import parse_incr

from tjunlp.common.checks import ConfigurationError
from tjunlp.common.config import Config
from tjunlp.core.instance import Instance
from tjunlp.core.vocabulary import Vocabulary

logger = logging.getLogger(__name__)


def _is_divider(line: str) -> bool:
    return line.strip() == ""


class DataReader(object):
    """

    """

    def __init__(self,
                 train_file: str = None,
                 dev_file: str = None,
                 test_file: str = None,
                 data_dir: str = None,
                 **kwargs):
        self.train_path = os.path.join(data_dir, train_file)
        self.dev_path = os.path.join(data_dir, dev_file)
        self.test_path = os.path.join(data_dir, test_file)

    def read(self, file_path: str) -> Iterable[Instance]:
        """
        TODO(izhx): 加入缓存
        :param file_path:
        :return:
        """
        if os.path.exists(file_path):
            data = self._read(file_path)
            if not data:
                raise ConfigurationError(f"No data at: {file_path}")
            return data
        else:
            raise ConfigurationError(f"File not exist! Please check the path: {file_path}")

    def read_train(self) -> Iterable[Instance]:
        return self.read(self.train_path)

    def read_dev(self) -> Iterable[Instance]:
        return self.read(self.dev_path)

    def read_test(self) -> Iterable[Instance]:
        return self.read(self.test_path)

    def _read(self, file_path: str) -> Iterable[Instance]:
        raise NotImplementedError

    def text_to_instance(self) -> Instance:
        raise NotImplementedError

    def instance_to_index(self, instance: Instance) -> Instance:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: Config) -> 'DataReader':
        if 'data' in config:
            return cls(**config['data'])
        else:
            raise ConfigurationError("Check your configuration file!")


class ConlluReader(DataReader):
    """

    """

    def __init__(self, multi_lang,
                 use_language_specific_pos: bool = False,
                 tokenizer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.multi_lang = multi_lang
        self.use_language_specific_pos = use_language_specific_pos
        self.tokenizer = tokenizer

    @overrides
    def _read(self, file_path: str, lang: str = 'en') -> Iterable[Instance]:
        with open(file_path, "r") as conllu_file:
            logger.info("Reading UD instances from conllu dataset at: %s", file_path)

            for annotation in parse_incr(conllu_file):
                # CoNLLU annotations sometimes add back in words that have been elided
                # in the original sentence; we remove these, as we're just predicting
                # dependencies for the original sentence.
                # We filter by None here as elided words have a non-integer word id,
                # and are replaced with None by the conllu python library.
                annotation = [x for x in annotation if isinstance(x["id"], int)]

                heads = [x["head"] for x in annotation]
                rels = [x["deprel"] for x in annotation]
                if self.multi_lang:
                    words = [f'{x["form"]}_{x["lemma"]}' for x in annotation]
                else:
                    words = [x["form"] for x in annotation]
                if self.use_language_specific_pos:
                    pos_tags = [x["xpostag"] for x in annotation]
                else:
                    pos_tags = [x["upostag"] for x in annotation]
                yield self.text_to_instance(lang, words, pos_tags, (rels, heads))

    @overrides
    def text_to_instance(
            self,
            lang: str,
            words: List[str],
            upos_tags: List[str],
            dependencies: Tuple = None,
    ) -> Instance:

        """
        TODO(izhx) 加一个虚根？

        Parameters
        ----------
        lang : ``str``, required.
        words : ``List[str]``, required.
            The words in the sentence to be encoded.
        upos_tags : ``List[str]``, required.
            The universal dependencies POS tags for each word.
        dependencies : ``List[Tuple[str, int]]``, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        Returns
        -------
        An instance containing words, upos tags, dependency head tags and head
        indices as fields.
        """
        fields: Dict[str, object] = {}

        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(" ".join(words))
        else:
            tokens = [t.lower() for t in words]

        fields["words"] = tokens
        fields["pos_tags"] = upos_tags
        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["rels"], fields["heads"] = dependencies

        fields["metadata"] = {"words": words, "pos": upos_tags, "lang": lang}
        return Instance(fields)

    @overrides
    def instance_to_index(self,
                          instance: Instance,
                          vocab: Vocabulary,
                          in_place: bool = True) -> Instance:
        if in_place:
            instance['words'] = vocab.tokens_to_indices(instance['words'], 'words')
            instance['pos_tags'] = vocab.tokens_to_indices(instance['pos_tags'], 'pos_tags')
            instance['rels'] = vocab.tokens_to_indices(instance['rels'], 'rels')
        else:
            instance['word_ids'] = vocab.tokens_to_indices(instance['words'], 'words')
            instance['pos_tag_ids'] = vocab.tokens_to_indices(instance['pos_tags'], 'pos_tags')
            instance['rel_ids'] = vocab.tokens_to_indices(instance['rels'], 'rels')
        return instance
