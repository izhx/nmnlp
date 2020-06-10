from typing import Dict, List, Tuple

from .model import Model

KEY_NAME, KEY_LR, KEY_PARAMS, KEY_OTHER = 'name', 'lr', 'params', 'other'


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
    for group in optimizer.param_groups:
        if KEY_NAME in group:
            yield f"{KEY_LR}_{group[KEY_NAME]}", group[KEY_LR]
        else:
            yield KEY_LR, group[KEY_LR]


def noam_lambda(model_size: int, warmup_steps: int, factor: float = 1.0):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first `warmup_steps` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    # Parameters
    model_size : `int`, required.
        The hidden size parameter which dominates the number of parameters in your model.
    warmup_steps : `int`, required.
        The number of steps to linearly increase the learning rate.
    factor : `float`, optional (default = 1.0).
        The overall scale factor for the learning rate decay.
    """
    factor = factor * model_size ** (-0.5)
    warm = warmup_steps ** (-1.5)

    def noam(step) -> float:
        step = 1 if step < 1 else step
        scale = factor * min(step ** (-0.5), step * warm)
        return scale

    return noam
