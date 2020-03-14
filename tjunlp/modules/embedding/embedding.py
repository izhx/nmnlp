"""
Code from allen.
"""

import io
import re
import logging
import itertools
from typing import Optional, Tuple, Sequence, cast, IO, Iterator, Any, NamedTuple

import numpy
import torch
from torch.nn.functional import embedding

from tjunlp.common.tqdm import Tqdm
from tjunlp.common.config import Config
from tjunlp.common.checks import ConfigurationError
from tjunlp.core.vocabulary import Vocabulary
from tjunlp.common.file_utils import get_file_extension, cached_path, is_url_or_existing_file
from tjunlp.modules import util

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Embedding(torch.nn.Module):
    """
    A more featureful embedding module than the default in Pytorch.  Adds the ability to:

        1. embed higher-order inputs
        2. pre-specify the weight matrix
        3. use a non-trainable embedding
        4. project the resultant embeddings to some other dimension (which only makes sense with
           non-trainable embeddings).
        5. build all of this easily ``from_params``

    Parameters
    ----------
    num_embeddings : int
        Size of the dictionary of embeddings (vocabulary size).
    embedding_dim : int
        The size of each embedding vector.
    weight : torch.FloatTensor, (optional, default=None)
        A pre-initialised weight matrix for the embedding lookup, allowing the use of
        pretrained vectors.
    padding_index : int, (optional, default=None)
        If given, pads the output with zeros whenever it encounters the index.
    trainable : bool, (optional, default=True)
        Whether or not to optimize the embedding parameters.
    max_norm : float, (optional, default=None)
        If given, will renormalize the embeddings to always have a norm lesser than this
    norm_type : float, (optional, default=2)
        The p of the p-norm to compute for the max_norm option
    scale_grad_by_freq : boolean, (optional, default=False)
        If given, this will scale gradients by the frequency of the words in the mini-batch.
    sparse : bool, (optional, default=False)
        Whether or not the Pytorch backend should use a sparse representation of the embedding weight.
    vocab_namespace : str, (optional, default=None)
        In case of fine-tuning/transfer learning, the model's embedding matrix needs to be
        extended according to the size of extended-vocabulary. To be able to know how much to
        extend the embedding-matrix, it's necessary to know which vocab_namspace was used to
        construct it in the original training. We store vocab_namespace used during the original
        training as an attribute, so that it can be retrieved during fine-tuning.
    pretrained_file : str, (optional, default=None)
        Used to keep track of what is the source of the weights and loading more embeddings at test time.
        **It does not load the weights from this pretrained_file.** For that purpose, use
        ``Embedding.from_params``.

    Returns
    -------
    An Embedding module.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 weight: torch.FloatTensor = None,
                 padding_index: int = None,
                 trainable: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 vocab_namespace: str = None,
                 pretrained_file: str = None) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self._vocab_namespace = vocab_namespace
        self._pretrained_file = pretrained_file
        self.output_dim = embedding_dim

        if weight is None:
            weight = torch.FloatTensor(num_embeddings, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            if weight.size() != (num_embeddings, embedding_dim):
                raise ConfigurationError("A weight matrix was passed with contradictory embedding shapes.")
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)

        if self.padding_index is not None:
            self.weight.data[self.padding_index].fill_(0)

    def forward(self, inputs):  # pylint: disable=arguments-differ
        # inputs may have extra dimensions (batch_size, d1, ..., dn, sequence_length),
        # but embedding expects (batch_size, sequence_length), so pass inputs to
        # util.combine_initial_dims (which is a no-op if there are no extra dimensions).
        # Remember the original size.
        original_size = inputs.size()
        inputs = util.combine_initial_dims(inputs)

        embedded = embedding(inputs, self.weight,
                             padding_idx=self.padding_index,
                             max_norm=self.max_norm,
                             norm_type=self.norm_type,
                             scale_grad_by_freq=self.scale_grad_by_freq,
                             sparse=self.sparse)

        # Now (if necessary) add back in the extra dimensions.
        embedded = util.uncombine_initial_dims(embedded, original_size)
        return embedded

    def extend_vocab(self,  # pylint: disable=arguments-differ
                     extended_vocab: Vocabulary,
                     vocab_namespace: str = None,
                     extension_pretrained_file: str = None,
                     model_path: str = None):
        """
        Extends the embedding matrix according to the extended vocabulary.
        If extension_pretrained_file is available, it will be used for initializing the new words
        embeddings in the extended vocabulary; otherwise we will check if _pretrained_file attribute
        is already available. If none is available, they will be initialized with xavier uniform.

        Parameters
        ----------
        extended_vocab : Vocabulary:
            Vocabulary extended from original vocabulary used to construct
            this ``Embedding``.
        vocab_namespace : str, (optional, default=None)
            In case you know what vocab_namespace should be used for extension, you
            can pass it. If not passed, it will check if vocab_namespace used at the
            time of ``Embedding`` construction is available. If so, this namespace
            will be used or else extend_vocab will be a no-op.
        extension_pretrained_file : str, (optional, default=None)
            A file containing pretrained embeddings can be specified here. It can be
            the path to a local file or an URL of a (cached) remote file. Check format
            details in ``from_params`` of ``Embedding`` class.
        model_path : str, (optional, default=None)
            Path traversing the model attributes upto this embedding module.
            Eg. "_text_field_embedder.token_embedder_tokens". This is only useful
            to give helpful error message when extend_vocab is implicitly called
            by fine-tune or any other command.
        """
        # Caveat: For allennlp v0.8.1 and below, we weren't storing vocab_namespace as an attribute,
        # knowing which is necessary at time of embedding vocab extension. So old archive models are
        # currently unextendable.

        vocab_namespace = vocab_namespace or self._vocab_namespace
        if not vocab_namespace:
            # It's not safe to default to "tokens" or any other namespace.
            logging.info("Loading a model trained before embedding extension was implemented; "
                         "pass an explicit vocab namespace if you want to extend the vocabulary.")
            return

        extended_num_embeddings = extended_vocab.get_vocab_size(vocab_namespace)
        if extended_num_embeddings == self.num_embeddings:
            # It's already been extended. No need to initialize / read pretrained file in first place (no-op)
            return

        if extended_num_embeddings < self.num_embeddings:
            raise ConfigurationError(f"Size of namespace, {vocab_namespace} for extended_vocab is smaller than "
                                     f"embedding. You likely passed incorrect vocab or namespace for extension.")

        # Case 1: user passed extension_pretrained_file and it's available.
        if extension_pretrained_file and is_url_or_existing_file(extension_pretrained_file):
            # Don't have to do anything here, this is the happy case.
            pass
        # Case 2: user passed extension_pretrained_file and it's not available
        elif extension_pretrained_file:
            raise ConfigurationError(f"You passed pretrained embedding file {extension_pretrained_file} "
                                     f"for model_path {model_path} but it's not available.")
        # Case 3: user didn't pass extension_pretrained_file, but pretrained_file attribute was
        # saved during training and is available.
        elif is_url_or_existing_file(self._pretrained_file):
            extension_pretrained_file = self._pretrained_file
        # Case 4: no file is available, hope that pretrained embeddings weren't used in the first place and warn
        else:
            extra_info = (f"Originally pretrained_file was at "
                          f"{self._pretrained_file}. " if self._pretrained_file else "")
            # It's better to warn here and not give error because there is no way to distinguish between
            # whether pretrained-file wasn't used during training or user forgot to pass / passed incorrect
            # mapping. Raising an error would prevent fine-tuning in the former case.
            logging.warning(f"Embedding at model_path, {model_path} cannot locate the pretrained_file. "
                            f"{extra_info} If you are fine-tuning and want to use using pretrained_file for "
                            f"embedding extension, please pass the mapping by --embedding-sources argument.")

        embedding_dim = self.weight.data.shape[-1]
        if not extension_pretrained_file:
            extra_num_embeddings = extended_num_embeddings - self.num_embeddings
            extra_weight = torch.FloatTensor(extra_num_embeddings, embedding_dim)
            torch.nn.init.xavier_uniform_(extra_weight)
        else:
            # It's easiest to just reload the embeddings for the entire vocab,
            # then only keep the ones we need.
            whole_weight = _read_embeddings_from_text_file(extension_pretrained_file, embedding_dim,
                                                           extended_vocab, vocab_namespace)
            extra_weight = whole_weight[self.num_embeddings:, :]

        device = self.weight.data.device
        extended_weight = torch.cat([self.weight.data, extra_weight.to(device)], dim=0)
        self.weight = torch.nn.Parameter(extended_weight, requires_grad=self.weight.requires_grad)

    # Custom logic requires custom from_params.
    @classmethod
    def from_config(cls, vocab: Vocabulary, config: Config) -> 'Embedding':  # type: ignore
        """
        We need the vocabulary here to know how many items we need to embed, and we look for a
        ``vocab_namespace`` key in the parameter dictionary to know which vocabulary to use.  If
        you know beforehand exactly how many embeddings you need, or aren't using a vocabulary
        mapping for the things getting embedded here, then you can pass in the ``num_embeddings``
        key directly, and the vocabulary will be ignored.

        In the configuration file, a file containing pretrained embeddings can be specified
        using the parameter ``"pretrained_file"``.
        It can be the path to a local file.
        Format:

            * text file - an utf-8 encoded text file with space separated fields::

                    [word] [dim 1] [dim 2] ...

              The text file can eventually be compressed with gzip, bz2, lzma or zip.

        """
        # pylint: disable=arguments-differ
        num_embeddings = config.get('num_embeddings', None)
        # If num_embeddings is present, set default namespace to None so that extend_vocab
        # call doesn't misinterpret that some namespace was originally used.
        vocab_namespace = config.get("vocab_namespace", None if num_embeddings else "tokens")
        if num_embeddings is None:
            num_embeddings = vocab.get_vocab_size(vocab_namespace)
        embedding_dim = config.get('embedding_dim')
        pretrained_file = config.get("pretrained_file", None)
        trainable = config.get("trainable", True)
        padding_index = config.get('padding_index', None)
        max_norm = config.get('max_norm', None)
        norm_type = config.get('norm_type', 2.)
        scale_grad_by_freq = config.get('scale_grad_by_freq', False)
        sparse = config.get('sparse', False)

        if pretrained_file:
            # If we're loading a saved model, we don't want to actually read a pre-trained
            # embedding file - the embeddings will just be in our saved weights, and we might not
            # have the original embedding file anymore, anyway.
            weight = _read_embeddings_from_text_file(pretrained_file,
                                                     embedding_dim,
                                                     vocab,
                                                     vocab_namespace)
        else:
            weight = None

        return cls(num_embeddings=num_embeddings,
                   embedding_dim=embedding_dim,
                   weight=weight,
                   padding_index=padding_index,
                   trainable=trainable,
                   max_norm=max_norm,
                   norm_type=norm_type,
                   scale_grad_by_freq=scale_grad_by_freq,
                   sparse=sparse,
                   vocab_namespace=vocab_namespace)


def _read_embeddings_from_text_file(file_uri: str,
                                    embedding_dim: int,
                                    vocab: Vocabulary,
                                    namespace: str = "tokens") -> torch.FloatTensor:
    """
    Read pre-trained word vectors from an eventually compressed text file, possibly contained
    inside an archive with multiple files. The text file is assumed to be utf-8 encoded with
    space-separated fields: [word] [dim 1] [dim 2] ...

    Lines that contain more numerical tokens than ``embedding_dim`` raise a warning and are skipped.

    The remainder of the docstring is identical to ``_read_pretrained_embeddings_file``.
    """
    tokens_to_keep = set(vocab.get_index_to_token_vocabulary(namespace).values())
    vocab_size = vocab.get_vocab_size(namespace)
    embeddings = {}

    # First we read the embeddings from the file, only keeping vectors for the words we need.
    logger.info("Reading pretrained embeddings from file")

    with EmbeddingsTextFile(file_uri) as embeddings_file:
        for line in Tqdm.tqdm(embeddings_file):
            token = line.split(' ', 1)[0]
            if token in tokens_to_keep:
                fields = line.rstrip().split(' ')
                if len(fields) - 1 != embedding_dim:
                    # Sometimes there are funny unicode parsing problems that lead to different
                    # fields lengths (e.g., a word with a unicode space character that splits
                    # into more than one column).  We skip those lines.  Note that if you have
                    # some kind of long header, this could result in all of your lines getting
                    # skipped.  It's hard to check for that here; you just have to look in the
                    # embedding_misses_file and at the model summary to make sure things look
                    # like they are supposed to.
                    logger.warning("Found line with wrong number of dimensions (expected: %d; actual: %d): %s",
                                   embedding_dim, len(fields) - 1, line)
                    continue

                vector = numpy.asarray(fields[1:], dtype='float32')
                embeddings[token] = vector

    if not embeddings:
        raise ConfigurationError("No embeddings of correct dimension found; you probably "
                                 "misspecified your embedding_dim parameter, or didn't "
                                 "pre-populate your Vocabulary")

    all_embeddings = numpy.asarray(list(embeddings.values()))
    embeddings_mean = float(numpy.mean(all_embeddings))
    embeddings_std = float(numpy.std(all_embeddings))
    # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
    # then filling in the word vectors we just read.
    logger.info("Initializing pre-trained embedding layer")
    embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean,
                                                                            embeddings_std)
    num_tokens_found = 0
    index_to_token = vocab.get_index_to_token_vocabulary(namespace)
    for i in range(vocab_size):
        token = index_to_token[i]

        # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
        # so the word has a random initialization.
        if token in embeddings:
            embedding_matrix[i] = torch.FloatTensor(embeddings[token])
            num_tokens_found += 1
        else:
            logger.debug("Token %s was not found in the embedding file. Initialising randomly.", token)

    logger.info("Pretrained embeddings were found for %d out of %d tokens",
                num_tokens_found, vocab_size)

    return embedding_matrix


class EmbeddingsTextFile(Iterator[str]):
    """
    Utility class for opening embeddings text files. Handles various compression formats,
    as well as context management.

    Parameters
    ----------
    file_uri: a file system path or a URL of an eventually compressed text file
    encoding: str
    """
    DEFAULT_ENCODING = 'utf-8'

    def __init__(self,
                 file_uri: str,
                 encoding: str = DEFAULT_ENCODING) -> None:
        # All the python packages for compressed files share the same interface of io.open
        extension = get_file_extension(file_uri)

        # Some systems don't have support for all of these libraries, so we import them only
        # when necessary.
        package = None
        if extension in ['.txt', '.vec']:
            package = io
        elif extension == '.gz':
            import gzip
            package = gzip
        elif extension == ".bz2":
            import bz2
            package = bz2
        elif extension == ".lzma":
            import lzma
            package = lzma

        if package is None:
            logger.warning('The embeddings file has an unknown file extension "%s". '
                           'We will assume the file is an (uncompressed) text file', extension)
            package = io

        self._handle = package.open(file_uri, 'rt', encoding=encoding)  # type: ignore

        # To use this with tqdm we'd like to know the number of tokens. It's possible that the
        # first line of the embeddings file contains this: if it does, we want to start iteration
        # from the 2nd line, otherwise we want to start from the 1st.
        # Unfortunately, once we read the first line, we cannot move back the file iterator
        # because the underlying file may be "not seekable"; we use itertools.chain instead.
        first_line = next(self._handle)  # this moves the iterator forward
        self.num_tokens = EmbeddingsTextFile._get_num_tokens_from_first_line(first_line)
        if self.num_tokens:
            # the first line is a header line: start iterating from the 2nd line
            self._iterator = self._handle
        else:
            # the first line is not a header line: start iterating from the 1st line
            self._iterator = itertools.chain([first_line], self._handle)

    def read(self) -> str:
        return ''.join(self._iterator)

    def readline(self) -> str:
        return next(self._iterator)

    def __enter__(self) -> 'EmbeddingsTextFile':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._handle.close()

    def __iter__(self) -> 'EmbeddingsTextFile':
        return self

    def __next__(self) -> str:
        return next(self._iterator)

    def __len__(self) -> Optional[int]:
        """ Hack for tqdm: no need for explicitly passing ``total=file.num_tokens`` """
        if self.num_tokens:
            return self.num_tokens
        raise AttributeError('an object of type EmbeddingsTextFile has "len()" only if the underlying '
                             'text file declares the number of tokens (i.e. the number of lines following)'
                             'in the first line. That is not the case of this particular instance.')

    @staticmethod
    def _get_num_tokens_from_first_line(line: str) -> Optional[int]:
        """ This function takes in input a string and if it contains 1 or 2 integers, it assumes the
        largest one it the number of tokens. Returns None if the line doesn't match that pattern. """
        fields = line.split(' ')
        if 1 <= len(fields) <= 2:
            try:
                int_fields = [int(x) for x in fields]
            except ValueError:
                return None
            else:
                num_tokens = max(int_fields)
                logger.info('Recognized a header line in the embedding file with number of tokens: %d',
                            num_tokens)
                return num_tokens
        return None
