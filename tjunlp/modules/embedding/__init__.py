from .bert_embedding import BertEmbedding
from .embedding import Embedding
from .packed_embedding import DeepEmbedding, PreTrainEmbedding


def build_word_embedding(name_or_path, freeze: str = 'all', **kwargs):
    """
    All embedding class should have attr `output_dim`.
    """
    if 'bert' in name_or_path:
        return BertEmbedding(name_or_path, freeze, **kwargs)
    if name_or_path == 'pretrain':
        return PreTrainEmbedding(**kwargs)
    if name_or_path == 'deep':
        return DeepEmbedding(**kwargs)
    if name_or_path == 'plain':
        return Embedding(**kwargs)
    raise ValueError("Wrong embedding name or path!")
