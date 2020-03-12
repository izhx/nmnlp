
from .bert_embedding import BertEmbedding
from .deep_embedding import DeepEmbedding


def build_word_embedding(name_or_path, freeze: str = 'all', **kwargs):
    """
    All embedding class should have attr `output_dim`.
    """
    if 'bert' in name_or_path:
        return BertEmbedding(name_or_path, freeze, **kwargs)
    if name_or_path == 'plain':
        return DeepEmbedding(**kwargs)
