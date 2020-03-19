"""
Code modified from allen.
"""

import io
import logging
import itertools
from typing import Optional, Tuple, Iterator, Any

import numpy
import torch
from torch.nn.functional import embedding

from tjunlp.common.checks import ConfigurationError
from tjunlp.common.util import output
from tjunlp.core import Vocabulary
from tjunlp.common.file_utils import get_file_extension
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
                 padding_index: int = 0,
                 trainable: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 **kwargs: Any) -> None:
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.padding_index = padding_index
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
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

    @classmethod
    def from_pretrain(cls,
                      vocab: Vocabulary,
                      pretrained_file: str,
                      vocab_namespace: str,
                      padding_index: int = 0,
                      trainable: bool = False,
                      max_norm: float = None,
                      norm_type: float = 2.,
                      scale_grad_by_freq: bool = False,
                      sparse: bool = False
                      ) -> 'Embedding':  # type: ignore
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
        # If we're loading a saved model, we don't want to actually read a pre-trained
        # embedding file - the embeddings will just be in our saved weights, and we might not
        # have the original embedding file anymore, anyway.

        tokens_to_keep = set(vocab.get_index_to_token_vocabulary(vocab_namespace).values())
        vocab_size = vocab.get_vocab_size(vocab_namespace)
        embeddings = dict()

        # First we read the embeddings from the file, only keeping vectors for the words we need.
        output("Reading pretrained embeddings from file")

        with EmbeddingsTextFile(pretrained_file) as embeddings_file:
            embedding_dim = embeddings_file.embedding_dim
            for line in embeddings_file:
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
        output("Initializing pre-trained embedding layer")
        embedding_matrix = torch.FloatTensor(vocab_size, embedding_dim).normal_(embeddings_mean,
                                                                                embeddings_std)
        num_tokens_found = 0
        index_to_token = vocab.get_index_to_token_vocabulary(vocab_namespace)
        for i in range(vocab_size):
            token = index_to_token[i]

            # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
            # so the word has a random initialization.
            if token in embeddings:
                embedding_matrix[i] = torch.FloatTensor(embeddings[token])
                num_tokens_found += 1
            else:
                logger.debug("Token %s was not found in the embedding file. Initialising randomly.", token)

        output(f"Pretrained embeddings were found for {num_tokens_found} out of {vocab_size} tokens")

        return cls(num_embeddings=embedding_matrix.size(0),
                   embedding_dim=embedding_matrix.size(1),
                   weight=embedding_matrix,
                   padding_index=padding_index,
                   trainable=trainable,
                   max_norm=max_norm,
                   norm_type=norm_type,
                   scale_grad_by_freq=scale_grad_by_freq,
                   sparse=sparse)


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
        self.num_tokens, self.embedding_dim = self._read_first_line(first_line)
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
    def _read_first_line(line: str) -> Optional[Tuple]:
        """ This function takes in input a string and if it contains 1 or 2 integers, it assumes the
        largest one it the number of tokens. Returns None if the line doesn't match that pattern. """
        fields = line.split(' ')
        if 1 <= len(fields) <= 2:
            try:
                int_fields = [int(x) for x in fields]
            except ValueError:
                return None, None
            else:
                num_tokens, embedding_dim = max(int_fields), min(int_fields)
                logger.info('Recognized a header line with number of tokens: %d',
                            num_tokens)
                return num_tokens, embedding_dim
        else:
            raise ValueError('Unrecognized header line!')
