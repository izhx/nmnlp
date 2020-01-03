
from .bert_embedding import BertEmbedding
from .deep_embedding import DeepEmbedding


def build_word_embedding(name, freeze: str = 'all', **kwargs):
    """
    All embedding class should have attr `output_dim`.
    """
    if 'bert' in name:
        return BertEmbedding(name, freeze, **kwargs)
    if name == 'plain':
        return DeepEmbedding(**kwargs)
