"""
Code modified from allennlp.

A Vocabulary maps strings to integers, allowing for strings to be mapped to an
out-of-vocabulary token.
"""

import copy
import logging
import pickle
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union
from itertools import chain

from tjunlp.common.checks import ConfigurationError
from tjunlp.common.constant import KEY_TRAIN, KEY_DEV, PRETRAIN_POSTFIX, DEFAULT_FIELD
from tjunlp.common.util import field_match, output

logger = logging.getLogger(__name__)

DEFAULT_NON_PADDED_FIELDS = ("*tags", "*labels")
DEFAULT_PADDING_TOKEN = "<pad>"
DEFAULT_PADDING_INDEX = 0
DEFAULT_OOV_TOKEN = "<unk>"
DEFAULT_OOV_INDEX = 1


class _FieldDependentDefaultDict(defaultdict):
    """
    This is a `defaultdict
    <https://docs.python.org/2/library/collections.html#collections.defaultdict>`_ where the
    default value is dependent on the key that is passed.

    We use "fields" in the :class:`Vocabulary` object to keep track of several different
    mappings from strings to integers, so that we have a consistent API for mapping words, tags,
    labels, characters, or whatever else you want, into integers.  The issue is that some of those
    fields (words and characters) should have integers reserved for padding and
    out-of-vocabulary tokens, while others (labels and tags) shouldn't.  This class allows you to
    specify filters on the field (the key used in the ``defaultdict``), and use different
    default values depending on whether the field passes the filter.

    To do filtering, we take a set of ``non_padded_fields``.  This is a set of strings
    that are either matched exactly against the keys, or treated as suffixes, if the
    string starts with ``*``.  In other words, if ``*tags`` is in ``non_padded_fields`` then
    ``passage_tags``, ``question_tags``, etc. (anything that ends with ``tags``) will have the
    ``non_padded`` default value.

    Parameters
    ----------
    non_padded_fields : ``Iterable[str]``
        A set / list / tuple of strings describing which fields are not padded.  If a field
        (key) is missing from this dictionary, we will use :func:`field_match` to see whether
        the field should be padded.  If the given field matches any of the strings in this
        list, we will use ``non_padded_function`` to initialize the value for that field, and
        we will use ``padded_function`` otherwise.
    padded_function : ``Callable[[], Any]``
        A zero-argument function to call to initialize a value for a field that `should` be
        padded.
    non_padded_function : ``Callable[[], Any]``
        A zero-argument function to call to initialize a value for a field that should `not` be
        padded.
    """

    def __init__(self,
                 non_padded_fields: Iterable[str],
                 padded_function: Callable[[], Any],
                 non_padded_function: Callable[[], Any]) -> None:
        self._non_padded_fields = set(non_padded_fields)
        self._padded_function = padded_function
        self._non_padded_function = non_padded_function
        super(_FieldDependentDefaultDict, self).__init__()

    def __missing__(self, key: str):
        if any(field_match(pattern, key) for pattern in self._non_padded_fields):
            value = self._non_padded_function()
        else:
            value = self._padded_function()
        dict.__setitem__(self, key, value)
        return value

    def add_non_padded_fields(self, non_padded_fields: Set[str]):
        # add non_padded_fields which weren't already present
        self._non_padded_fields.update(non_padded_fields)


class _TokenToIndexDefaultDict(_FieldDependentDefaultDict):
    def __init__(self, non_padded_fields: Set[str], padding_token: str, oov_token: str) -> None:
        super(_TokenToIndexDefaultDict, self).__init__(non_padded_fields,
                                                       lambda: {
                                                           padding_token: 0, oov_token: 1},
                                                       lambda: {})


class _IndexToTokenDefaultDict(_FieldDependentDefaultDict):
    def __init__(self, non_padded_fields: Set[str], padding_token: str, oov_token: str) -> None:
        super(_IndexToTokenDefaultDict, self).__init__(non_padded_fields,
                                                       lambda: {
                                                           0: padding_token, 1: oov_token},
                                                       lambda: {})


def _read_pretrained_tokens(embeddings_file: str) -> List[str]:
    # Moving this import to the top breaks everything (cycling import, I guess)
    from tjunlp.modules.embedding.embedding import EmbeddingsTextFile

    output(f'Reading pretrained tokens from: <{embeddings_file}>')
    tokens: List[str] = []
    with EmbeddingsTextFile(embeddings_file) as file:
        for line_number, line in enumerate(file, start=1):
            token_end = line.find(' ')
            if token_end >= 0:
                token = line[:token_end]
                tokens.append(token)
            else:
                line_begin = line[:20] + '...' if len(line) > 20 else line
                logger.warning('Skipping line number %d: %s',
                               line_number, line_begin)
    return tokens


class Vocabulary(object):
    def __init__(self,
                 counter: Dict[str, Dict[str, int]] = None,
                 min_count: Dict[str, int] = None,
                 max_vocab_size: Union[int, Dict[str, int]] = None,
                 non_padded_fields: Iterable[str] = DEFAULT_NON_PADDED_FIELDS,
                 pretrained_files: Optional[Dict[str, str]] = None,
                 only_include_pretrained_words: bool = False,
                 tokens_to_add: Dict[str, List[str]] = None,
                 padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
                 oov_token: Optional[str] = DEFAULT_OOV_TOKEN) -> None:
        self._padding_token = padding_token
        self._oov_token = oov_token
        self._non_padded_fields = set(non_padded_fields)
        self._token_to_index = _TokenToIndexDefaultDict(self._non_padded_fields,
                                                        self._padding_token,
                                                        self._oov_token)
        self._index_to_token = _IndexToTokenDefaultDict(self._non_padded_fields,
                                                        self._padding_token,
                                                        self._oov_token)
        self._retained_counter: Optional[Dict[str, Dict[str, int]]] = None
        # Made an empty vocabulary, now extend it.
        self._extend(counter,
                     min_count,
                     max_vocab_size,
                     non_padded_fields,
                     pretrained_files,
                     only_include_pretrained_words,
                     tokens_to_add)

    """
    A Vocabulary maps strings to integers, allowing for strings to be mapped to an
    out-of-vocabulary token.

    Vocabularies are fit to a particular dataset, which we use to decide which tokens are
    in-vocabulary.

    Vocabularies also allow for several different fields, so you can have separate indices for
    'a' as a word, and 'a' as a character, for instance, and so we can use this object to also map
    tag and label strings to indices, for a unified :class:`~.fields.field.Field` API.  Most of the
    methods on this class allow you to pass in a field; by default we use the 'tokens'
    field, and you can omit the field argument everywhere and just use the default.

    Parameters
    ----------
    counter : ``Dict[str, Dict[str, int]]``, optional (default=``None``)
        A collection of counts from which to initialize this vocabulary.  We will examine the
        counts and, together with the other parameters to this class, use them to decide which
        words are in-vocabulary.  If this is ``None``, we just won't initialize the vocabulary with
        anything.
    min_count : ``Dict[str, int]``, optional (default=None)
        When initializing the vocab from a counter, you can specify a minimum count, and every
        token with a count less than this will not be added to the dictionary.  These minimum
        counts are `field-specific`, so you can specify different minimums for labels versus
        words tokens, for example.  If a field does not have a key in the given dictionary, we
        will add all seen tokens to that field.
    max_vocab_size : ``Union[int, Dict[str, int]]``, optional (default=``None``)
        If you want to cap the number of tokens in your vocabulary, you can do so with this
        parameter.  If you specify a single integer, every field will have its vocabulary fixed
        to be no larger than this.  If you specify a dictionary, then each field in the
        ``counter`` can have a separate maximum vocabulary size.  Any missing key will have a value
        of ``None``, which means no cap on the vocabulary size.
    non_padded_fields : ``Iterable[str]``, optional
        By default, we assume you are mapping word / character tokens to integers, and so you want
        to reserve word indices for padding and out-of-vocabulary tokens.  However, if you are
        mapping NER or SRL tags, or class labels, to integers, you probably do not want to reserve
        indices for padding and out-of-vocabulary tokens.  Use this field to specify which
        fields should `not` have padding and OOV tokens added.

        The format of each element of this is either a string, which must match field names
        exactly,  or ``*`` followed by a string, which we match as a suffix against field names.

        We try to make the default here reasonable, so that you don't have to think about this.
        The default is ``("*tags", "*labels")``, so as long as your field ends in "tags" or
        "labels" (which is true by default for all tag and label fields in this code), you don't
        have to specify anything here.
    pretrained_files : ``Dict[str, str]``, optional
        If provided, this map specifies the path to optional pretrained embedding files for each
        field. This can be used to either restrict the vocabulary to only words which appear
        in this file, or to ensure that any words in this file are included in the vocabulary
        regardless of their count, depending on the value of ``only_include_pretrained_words``.
        Words which appear in the pretrained embedding file but not in the data are NOT included
        in the Vocabulary.
    min_pretrained_embeddings : ``Dict[str, int]``, optional
        If provided, specifies for each field a minimum number of lines (typically the
        most common words) to keep from pretrained embedding files, even for words not
        appearing in the data.
    only_include_pretrained_words : ``bool``, optional (default=False)
        This defines the strategy for using any pretrained embedding files which may have been
        specified in ``pretrained_files``. If False, an inclusive strategy is used: and words
        which are in the ``counter`` and in the pretrained file are added to the ``Vocabulary``,
        regardless of whether their count exceeds ``min_count`` or not. If True, we use an
        exclusive strategy: words are only included in the Vocabulary if they are in the pretrained
        embedding file (their count must still be at least ``min_count``).
    tokens_to_add : ``Dict[str, List[str]]``, optional (default=None)
        If given, this is a list of tokens to add to the vocabulary, keyed by the field to add
        the tokens to.  This is a way to be sure that certain items appear in your vocabulary,
        regardless of any other vocabulary computation.
    """

    def __getstate__(self):
        """
        Need to sanitize defaultdict and defaultdict-like objects
        by converting them to vanilla dicts when we pickle the vocabulary.
        """
        state = copy.copy(self.__dict__)
        state["_token_to_index"] = dict(state["_token_to_index"])
        state["_index_to_token"] = dict(state["_index_to_token"])

        if "_retained_counter" in state:
            state["_retained_counter"] = {key: dict(value)
                                          for key, value in state["_retained_counter"].items()}

        return state

    def __setstate__(self, state):
        """
        Conversely, when we unpickle, we need to reload the plain dicts
        into our special DefaultDict subclasses.
        """
        # pylint: disable=attribute-defined-outside-init
        self.__dict__ = copy.copy(state)
        self._token_to_index = _TokenToIndexDefaultDict(self._non_padded_fields,
                                                        self._padding_token,
                                                        self._oov_token)
        self._token_to_index.update(state["_token_to_index"])
        self._index_to_token = _IndexToTokenDefaultDict(self._non_padded_fields,
                                                        self._padding_token,
                                                        self._oov_token)
        self._index_to_token.update(state["_index_to_token"])

    def save(self, file_path: str) -> None:
        pickle.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str) -> 'Vocabulary':
        return pickle.load(file_path)

    @classmethod
    def from_instances(cls,
                       instances,
                       create_fields: Set[str],
                       min_count: Dict[str, int] = None,
                       max_vocab_size: Union[int, Dict[str, int]] = None,
                       non_padded_fields: Iterable[str] = DEFAULT_NON_PADDED_FIELDS,
                       pretrained_files: Optional[Dict[str, str]] = None,
                       only_include_pretrained_words: bool = False,
                       tokens_to_add: Dict[str, List[str]] = None,
                       padding_token: Optional[str] = DEFAULT_PADDING_TOKEN,
                       oov_token: Optional[str] = DEFAULT_OOV_TOKEN
                       ) -> 'Vocabulary':
        """
        Constructs a vocabulary given a collection of `Instances` and some parameters.
        We count all of the vocabulary items in the instances, then pass those counts
        and the other parameters, to :func:`__init__`.  See that method for a description
        of what the other parameters do.
        """
        output("Fitting token dictionary from dataset.")
        field_token_counts = defaultdict(lambda: defaultdict(int))
        if isinstance(instances, dict):
            if isinstance(instances[KEY_DEV], dict):
                instances = chain(
                    instances[KEY_TRAIN], *instances[KEY_DEV].values())
            elif isinstance(instances[KEY_DEV], list):
                instances = chain(instances[KEY_TRAIN], *instances[KEY_DEV])
            else:
                instances = instances[KEY_TRAIN] + instances[KEY_DEV]
        for instance in instances:
            for field in create_fields:
                for token in instance[field]:
                    field_token_counts[field][token] += 1

        return cls(counter=field_token_counts,
                   min_count=min_count,
                   max_vocab_size=max_vocab_size,
                   non_padded_fields=non_padded_fields,
                   pretrained_files=pretrained_files,
                   only_include_pretrained_words=only_include_pretrained_words,
                   tokens_to_add=tokens_to_add,
                   padding_token=padding_token,
                   oov_token=oov_token)

    def _extend(self,
                counter: Dict[str, Dict[str, int]] = None,
                min_count: Dict[str, int] = None,
                max_vocab_size: Union[int, Dict[str, int]] = None,
                non_padded_fields: Iterable[str] = DEFAULT_NON_PADDED_FIELDS,
                pretrained_files: Optional[Dict[str, str]] = None,
                only_include_pretrained_words: bool = False,
                tokens_to_add: Dict[str, List[str]] = None) -> None:
        """
        This method can be used for extending already generated vocabulary.
        It takes same parameters as Vocabulary initializer. The token_to_index
        and index_to_token mappings of calling vocabulary will be retained.
        It is an inplace operation so None will be returned.
        """
        if not isinstance(max_vocab_size, dict):
            int_max_vocab_size = max_vocab_size
            max_vocab_size = defaultdict(lambda: int_max_vocab_size)
        min_count = min_count or {}
        pretrained_files = pretrained_files or {}
        non_padded_fields = set(non_padded_fields)
        counter = counter or {}
        tokens_to_add = tokens_to_add or {}

        self._retained_counter = counter
        # Make sure vocabulary extension is safe.
        current_fields = {*self._token_to_index}
        extension_fields = {*counter, *tokens_to_add}

        for field in current_fields & extension_fields:
            # if new field was already present
            # Either both should be padded or none should be.
            original_padded = not any(field_match(pattern, field)
                                      for pattern in self._non_padded_fields)
            extension_padded = not any(field_match(pattern, field)
                                       for pattern in non_padded_fields)
            if original_padded != extension_padded:
                raise ConfigurationError("Common field {} has conflicting ".format(field) +
                                         "setting of padded = True/False. " +
                                         "Hence extension cannot be done.")

        # Add new non-padded fields for extension
        self._token_to_index.add_non_padded_fields(non_padded_fields)
        self._index_to_token.add_non_padded_fields(non_padded_fields)
        self._non_padded_fields.update(non_padded_fields)

        for field in counter:
            token_counts = list(counter[field].items())
            token_counts.sort(key=lambda x: x[1], reverse=True)
            try:
                max_vocab = max_vocab_size[field]
            except KeyError:
                max_vocab = None
            if max_vocab:
                token_counts = token_counts[:max_vocab]
            filed_min_count = min_count.get(field, 1)

            if field in pretrained_files:
                pretrained_set = set(_read_pretrained_tokens(pretrained_files[field]))
                if only_include_pretrained_words:
                    for token, count in token_counts:
                        if token in pretrained_set and count >= filed_min_count:
                            self.add_token_to_field(token, field)
                else:  # 分成两个字典
                    field_pretrained = field + PRETRAIN_POSTFIX
                    field_counter = {k: v for k, v in counter[field].items() if v > filed_min_count}
                    for token in pretrained_set:
                        self.add_token_to_field(token, field)
                        self.add_token_to_field(token, field_pretrained)
                        if token in field_counter:
                            field_counter.pop(token)
                    for token, count in field_counter.items():
                        self.add_token_to_field(token, field)
            else:
                for token, count in token_counts:
                    if count >= filed_min_count:
                        self.add_token_to_field(token, field)

        for field, tokens in tokens_to_add.items():
            for token in tokens:
                self.add_token_to_field(token, field)

    def fields(self):
        return {*self._token_to_index}

    def is_padded(self, field: str) -> bool:
        """
        Returns whether or not there are padding and OOV tokens added to the given field.
        """
        return self._index_to_token[field][0] == self._padding_token

    def add_token_to_field(self, token: str, field: str = DEFAULT_FIELD) -> int:
        """
        Adds ``token`` to the index, if it is not already present.  Either way, we return the index of
        the token.
        """
        if not isinstance(token, str):
            raise ValueError("Vocabulary tokens must be strings, or saving and loading will break."
                             "  Got %s (with type %s)" % (repr(token), type(token)))
        if token not in self._token_to_index[field]:
            index = len(self._token_to_index[field])
            self._token_to_index[field][token] = index
            self._index_to_token[field][index] = token
            return index
        else:
            return self._token_to_index[field][token]

    def add_tokens_to_field(self, tokens: List[str], field: str = DEFAULT_FIELD) -> List[int]:
        """
        Adds ``tokens`` to the index, if they are not already present.  Either way, we return the
        indices of the tokens in the order that they were given.
        """
        return [self.add_token_to_field(token, field) for token in tokens]

    def token_to_index(self, token: str, field: str = DEFAULT_FIELD) -> int:
        if token in self._token_to_index[field]:
            return self._token_to_index[field][token]
        else:
            try:
                return self._token_to_index[field][self._oov_token]
            except KeyError:
                logger.error('Field: %s', field)
                logger.error('Token: %s', token)
                raise

    def tokens_to_indices(self, tokens: List, field: str = DEFAULT_FIELD) -> List[int]:
        return [self.token_to_index(token, field) for token in tokens]

    def index_to_token(self, index: int, field: str = DEFAULT_FIELD) -> str:
        return self._index_to_token[field][index]

    def get_index_to_token_vocabulary(self, namespace: str = DEFAULT_FIELD) -> Dict[int, str]:
        return self._index_to_token[namespace]

    def get_token_to_index_vocabulary(self, namespace: str = DEFAULT_FIELD) -> Dict[str, int]:
        return self._token_to_index[namespace]

    def get_vocab_size(self, field: str = DEFAULT_FIELD) -> int:
        return len(self._token_to_index[field])

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __str__(self) -> str:
        base_string = f"Vocabulary with field:\n"
        non_padded_fields = f"\tNon Padded field: {self._non_padded_fields}\n"
        fields = [f"\tfield: {name}, Size: {self.get_vocab_size(name)} \n"
                  for name in self._index_to_token]
        return " ".join([base_string, non_padded_fields] + fields)

    def __repr__(self) -> str:
        # This is essentially the same as __str__, but with no newlines
        base_string = f"Vocabulary with fields: "
        fields = [f"{name}, Size: {self.get_vocab_size(name)} ||"
                  for name in self._index_to_token]
        non_padded_fields = f"Non Padded fields: {self._non_padded_fields}"
        return " ".join([base_string] + fields + [non_padded_fields])

    def __getitem__(self, item):
        return self._token_to_index[item]

    def print_statistics(self) -> None:
        if self._retained_counter:
            logger.info("Printed vocabulary statistics are only for the part of the vocabulary generated "
                        "from instances. If vocabulary is constructed by extending saved vocabulary with "
                        "dataset instances, the directly loaded portion won't be considered here.")
            print("\n\n----Vocabulary Statistics----\n")
            # Since we don't saved counter info, it is impossible to consider pre-saved portion.
            for field in self._retained_counter:
                tokens_with_counts = list(
                    self._retained_counter[field].items())
                tokens_with_counts.sort(key=lambda x: x[1], reverse=True)
                print(f"\nTop 10 most frequent tokens in field '{field}':")
                for token, freq in tokens_with_counts[:10]:
                    print(f"\tToken: {token}\t\tFrequency: {freq}")
                # Now sort by token length, not frequency
                tokens_with_counts.sort(key=lambda x: len(x[0]), reverse=True)

                print(f"\nTop 10 longest tokens in field '{field}':")
                for token, freq in tokens_with_counts[:10]:
                    print(
                        f"\tToken: {token}\t\tlength: {len(token)}\tFrequency: {freq}")

                print(f"\nTop 10 shortest tokens in field '{field}':")
                for token, freq in reversed(tokens_with_counts[-10:]):
                    print(
                        f"\tToken: {token}\t\tlength: {len(token)}\tFrequency: {freq}")
        else:
            # _retained_counter would be set only if instances were used for vocabulary construction.
            logger.info("Vocabulary statistics cannot be printed since "
                        "dataset instances were not used for its construction.")
