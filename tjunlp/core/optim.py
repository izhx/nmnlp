from typing import Dict, List, Tuple, Any

import torch
from torch.optim import Adam, SGD
from torch.optim.adamw import AdamW
from torch.optim import lr_scheduler

from transformers import AdamW as BertAdamW

from tjunlp.common.checks import ConfigurationError
from .model import Model

_OPTIMIZER = {
    'Adam': Adam,
    'SGD': SGD,
    'AdamW': AdamW,
    'BertAdamW': BertAdamW
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

KEY_NAME, KEY_LR, KEY_PARAMS, KEY_OTHER = 'name', 'lr', 'params', 'other'


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


def param_groups_with_different_lr(model: Model,
                                   default_lr: float,
                                   **kwargs: float) -> List[Dict]:
    """
    get param groups by keyword.
    """
    groups = {k: {KEY_PARAMS: list(), KEY_LR: kwargs[k], KEY_NAME: k} for k in kwargs}
    groups[KEY_OTHER] = {KEY_PARAMS: list(), KEY_LR: default_lr, KEY_NAME: KEY_OTHER}

    for name, param in model.named_parameters():
        for k in kwargs:
            if k in name:
                groups[k][KEY_PARAMS].append(param)
                break
        else:
            groups[KEY_OTHER][KEY_PARAMS].append(param)

    return list(groups.values())


def get_lrs(optimizer) -> Tuple[str, float]:
    for param_dict in optimizer.param_groups:
        if KEY_NAME in param_dict:
            yield f"{KEY_LR}_{param_dict[KEY_NAME]}", param_dict[KEY_LR]
        else:
            yield KEY_LR, param_dict[KEY_LR]
