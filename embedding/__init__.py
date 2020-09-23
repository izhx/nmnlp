from .bert_embedding import BertEmbedding
from .dep_embedding import DepSAWR
from .embedding import Embedding
from .packed_embedding import DeepEmbedding, PreTrainEmbedding


def build_word_embedding(name_or_path, freeze: str = 'all', **kwargs):
    """
    All embedding class should have attr `output_dim`.
    """
    if 'bert' in name_or_path:
        return BertEmbedding(name_or_path, freeze, **kwargs)
    elif 'elmo' in name_or_path:
        from .elmo_embedding import ElmoEmbedding
        return ElmoEmbedding(name_or_path, **kwargs)
    if name_or_path == 'pretrain':
        return PreTrainEmbedding(**kwargs)
    if name_or_path == 'deep':
        return DeepEmbedding(**kwargs)
    if name_or_path == 'plain':
        return Embedding(**kwargs)
    raise ValueError("Wrong embedding name or path!")
