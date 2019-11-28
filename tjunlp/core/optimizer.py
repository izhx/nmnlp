"""
Code from fastNLP.
"""

__all__ = [
    "Optimizer",
    "SGD",
    "Adam",
]
from typing import Dict

import torch.nn
from torch.optim.adamw import AdamW

from tjunlp.common.checks import ConfigurationError

_OPTIM = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
    'adamw': AdamW
}


class Optimizer(object):
    """
    Optimizer
    """

    def __init__(self, model_params, **kwargs):
        """
        :param model_params: a generator. E.g. ``model.parameters()`` for PyTorch models.
        :param kwargs: additional parameters.
        """
        if model_params is not None and not hasattr(model_params, "__next__"):
            raise RuntimeError("model parameters should be a generator, rather than {}.".format(type(model_params)))
        self.model_params = model_params
        self.settings = kwargs

    @staticmethod
    def _get_require_grads_param(params):
        """
        将params中不需要gradient的删除

        :param iterable params: parameters
        :return: list(nn.Parameters)
        """
        return [param for param in params if param.requires_grad]

    @classmethod
    def from_config(cls, model_params: Dict, setting: Dict):
        name = setting.pop('name')
        try:
            return _OPTIM[name](cls._get_require_grads_param(model_params), **setting)
        except:
            raise ConfigurationError(f'Wrong optimizer name: {name} !')


class NullOptimizer(Optimizer):
    """
    当不希望Trainer更新optimizer时，传入本optimizer，但请确保通过callback的方式对参数进行了更新。

    """

    def __init__(self):
        super().__init__(None)

    def __getattr__(self, item):
        def pass_func(*args, **kwargs):
            pass

        return pass_func


_OPTIM['null'] = NullOptimizer  # fix reference error


class SGD(Optimizer):
    """
    SGD
    """

    def __init__(self, lr=0.001, momentum=0, model_params=None):
        """
        :param float lr: learning rate. Default: 0.01
        :param float momentum: momentum. Default: 0
        :param model_params: a generator. E.g. ``model.parameters()`` for PyTorch models.
        """
        if not isinstance(lr, float):
            raise TypeError("learning rate has to be float.")
        super(SGD, self).__init__(model_params, lr=lr, momentum=momentum)

    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            # careful! generator cannot be assigned.
            return torch.optim.SGD(self._get_require_grads_param(model_params), **self.settings)
        else:
            return torch.optim.SGD(self._get_require_grads_param(self.model_params), **self.settings)


class Adam(Optimizer):
    """
    Adam
    """

    def __init__(self, lr=0.001, weight_decay=0, betas=(0.9, 0.999), eps=1e-8, amsgrad=False, model_params=None):
        """

        :param float lr: learning rate
        :param float weight_decay:
        :param eps:
        :param amsgrad:
        :param model_params: a generator. E.g. ``model.parameters()`` for PyTorch models.
        """
        if not isinstance(lr, float):
            raise TypeError("learning rate has to be float.")
        super(Adam, self).__init__(model_params, lr=lr, betas=betas, eps=eps, amsgrad=amsgrad,
                                   weight_decay=weight_decay)

    def construct_from_pytorch(self, model_params):
        if self.model_params is None:
            # careful! generator cannot be assigned.
            return torch.optim.Adam(self._get_require_grads_param(model_params), **self.settings)
        else:
            return torch.optim.Adam(self._get_require_grads_param(self.model_params), **self.settings)
