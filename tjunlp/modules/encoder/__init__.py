"""
a
"""
from .lstm import LstmEncoder


def build_encoder(input_size: int, name: str, **kwargs):
    """
    build encoder
    """
    if name == 'lstm':
        return LstmEncoder(input_size, bidirectional=False, **kwargs)
    elif name == 'bilstm':
        return LstmEncoder(input_size, bidirectional=True, **kwargs)
