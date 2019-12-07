import torch

from tjunlp.common.checks import ConfigurationError

_CRITERION = {
    'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
}


def build_loss(name, **kwargs):
    if name in _CRITERION:
        return _CRITERION[name](**kwargs)
    else:
        raise ConfigurationError(f'Wrong criterion name: {name} !')
