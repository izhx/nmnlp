from typing import Dict

import torch
from torch.optim.adamw import AdamW

from tjunlp.common.checks import ConfigurationError

_OPTIMIZER = {
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD,
    'AdamW': AdamW
}


def require_grads_param(params):
    """
    将params中不需要gradient的删除

    :param iterable params: parameters
    :return: list(nn.Parameters)
    """
    return [param for param in params if param.requires_grad]


def build_optimizer(model_params: Dict, name: str, **kwargs):
    if name in _OPTIMIZER:
        return _OPTIMIZER[name](require_grads_param(model_params), **kwargs)
    else:
        raise ConfigurationError(f'Wrong optimizer name: {name} !')
