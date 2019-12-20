
from .bert_embedding import PackedBertEmbedding
from .deep_embedding import DeepEmbedding


def build_word_embedding(name, freeze: str = 'all', **kwargs):
    """
    All embedding class should have attr `output_dim`.
    """
    if 'bert' in name:
        return PackedBertEmbedding(name, freeze)
    if name == 'plain':
        return DeepEmbedding(**kwargs)
