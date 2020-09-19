"""
为tensorboard，fitlog和自有工具等提供统一接口。
"""

import os
import shutil
import warnings
from enum import Enum, unique
from typing import Dict

from .util import now, output


@unique
class Backend(Enum):
    tensorboard = 1
    logviewer = 2
    fitlog = 3
    null = 4


class Writer(object):
    """
    接口风格与tensorboard相仿。
    """

    def __init__(self, log_dir: str, comment: str = '', backend: str = None, hyper_params: Dict = None):
        if not os.path.exists(os.path.abspath(log_dir)):
            os.mkdir(log_dir)
        self.log_dir = f"{log_dir}/{comment}_{now()}/"
        if os.path.exists(self.log_dir):  # 一般都没有
            shutil.rmtree(self.log_dir)
        os.mkdir(self.log_dir)
        output(f'Log dir <{self.log_dir}>')

        if backend in ('logviewer', 'lv', Backend.logviewer.value):
            self.type = Backend.logviewer
        elif backend in ('tensorboard', 'tb', Backend.tensorboard.value):
            self.type = Backend.tensorboard
            from torch.utils.tensorboard import SummaryWriter
            self.backend: SummaryWriter = SummaryWriter(self.log_dir)
        elif backend in ('fitlog', 'fl', 'fit', Backend.fitlog.value):
            raise NotImplementedError("暂不支持")
        else:
            self.type = Backend.null
            self.backend = None

    def add_params(self, hyper_params: Dict):
        warnings.warn("...")

    def add_scalar(self, tag, scalar_value, global_step):
        if self.type in (Backend.tensorboard, Backend.logviewer):
            self.backend.add_scalar(tag, scalar_value, global_step)
            self.backend.flush()
        if self.type == Backend.fitlog:
            warnings.warn("...")

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict,
                    global_step: int, key_prefix=None):
        for key, value in tag_scalar_dict.items():
            key = f'{key_prefix}_{key}' if key_prefix else key
            self.add_scalar(f"{main_tag}/{key}", value, global_step)

    def close(self):
        if self.type == Backend.tensorboard:
            self.backend.close()

    def flush(self):
        if self.type == Backend.tensorboard:
            self.backend.flush()
