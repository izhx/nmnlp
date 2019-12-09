from typing import Dict

import torch
from torch.optim import Adam, SGD
from torch.optim.adamw import AdamW
from torch.optim import lr_scheduler

from tjunlp.common.checks import ConfigurationError

_OPTIMIZER = {
    'Adam': Adam,
    'SGD': SGD,
    'AdamW': AdamW,
}

_SCHEDULER = {
    'LambdaLR': lr_scheduler.LambdaLR,
    'StepLR': lr_scheduler.StepLR,
    'MultiStepLR': lr_scheduler.MultiStepLR,
    'ExponentialLR': lr_scheduler.ExponentialLR,
    'CosineAnnealingLR': lr_scheduler.CosineAnnealingLR,
    'CyclicLR': lr_scheduler.CyclicLR,
    'CosineAnnealingWarmRestarts': lr_scheduler.CosineAnnealingWarmRestarts,
    # 'OneCycleLR':
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


def build_lr_scheduler(optimizer, name: str, **kwargs):
    if name in _SCHEDULER:
        return _SCHEDULER[name](optimizer, **kwargs)
    else:
        raise ConfigurationError(f'Wrong lr scheduler name: {name} !')