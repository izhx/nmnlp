from typing import Dict, Any, List
import os
import time
from datetime import datetime

import torch
# import torch.nn as nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from tjunlp.common.checks import ConfigurationError
from tjunlp.common.config import Config
from tjunlp.common.tqdm import Tqdm
from tjunlp.common.util import sys_info, sec_to_time
from tjunlp.core.dataset import DataSet
from tjunlp.core.model import Model
from tjunlp.core.vocabulary import Vocabulary
from tjunlp.data import KEY_TRAIN, KEY_DEV

EARLY_STOP_THRESHOLD = 5

SAVE_STRATEGY_ALL = 'all'
SAVE_STRATEGY_BEST = 'best'

DEFAULT_SAVE_DIR = './exp/'
DEFAULT_LOG_DIR = './tblog/'
DEFAULT_PREFIX = 'model_'

DEVICE_CPU = 'cpu'
DEVICE_CUDA = 'cuda'


def to_device(data, device: torch.device):
    if isinstance(data, Dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, torch.Tensor):
        data = data.to(device)
    else:
        pass
    return data


def clip_grad_func(parameters, method: str, **kwargs):
    if method == 'norm':
        clip_grad_norm_(parameters, **kwargs)
    elif method == 'value':
        clip_grad_value_(parameters, **kwargs)
    else:
        raise ConfigurationError("Wrong gradient clip type!")


class Trainer(object):
    """

    """

    def __init__(self,
                 cfg: Config,
                 dataset: Dict[str, DataSet],
                 vocabulary: Vocabulary,
                 model: Model,
                 optimizer: Optimizer,
                 scheduler: Any = None,  # _LRScheduler is protected
                 device: str = DEVICE_CPU,
                 clip_grad: Dict = None,
                 early_stop: bool = True,
                 epoch_num: int = 100,
                 epoch_start: int = 0,
                 update_every: int = 1,
                 validate_every: int = 10,
                 validate_after: int = 30,
                 save_after: int = 30,
                 save_dir: str = DEFAULT_SAVE_DIR,
                 save_strategy: str = SAVE_STRATEGY_BEST,
                 log_batch: bool = False,
                 log_dir: str = DEFAULT_LOG_DIR,
                 log_interval: int = 10,
                 dev_on_cpu: bool = True,
                 prefix: str = DEFAULT_PREFIX,
                 **kwargs):
        self.cfg = cfg
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.kwargs = kwargs
        self.clip_grad = clip_grad
        self.early_stop = early_stop
        self.epoch_num = epoch_num
        self.epoch_start = epoch_start
        self.update_every = update_every  # 梯度累积的步数 i.e. accumulation_steps
        self.validate_every = validate_every
        self.validate_after = validate_after
        self.save_after = save_after
        self.save_dir = save_dir
        self.save_strategy = save_strategy
        self.log_batch = log_batch
        self.log_interval = log_interval  # log every X batches
        self.prefix = prefix
        if device == DEVICE_CUDA and not torch.cuda.is_available():
            raise ConfigurationError("No GPU found, please run at CPU!")
        self.device = torch.device(device)
        self.dev_device = torch.device(DEVICE_CPU) if dev_on_cpu else self.device
        self.time_epoch = 0
        self.time_eval = 0
        self.best_metric = None
        self.stop_counter = 0

        for key in (KEY_TRAIN, KEY_DEV):
            if not self.dataset[key].indexed:
                self.dataset[key].index_dataset(vocabulary)

        if not os.path.exists(os.path.abspath(log_dir)):
            os.mkdir(log_dir)
        path = f"{log_dir}/{prefix + str(datetime.now())[:16].replace(' ', '_')}"
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        os.mkdir(path)
        self.writer = SummaryWriter(log_dir=path)

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        return

    def time_left(self):
        self.time_eval = self.time_epoch if self.time_eval == 0 else self.time_eval
        time_left = (self.time_epoch + self.time_eval / self.validate_every
                     ) * (self.epoch_num - self.epoch_start)
        return time_left  # TODO 计算均值

    def train(self):
        train_loader = DataLoader(dataset=self.dataset[KEY_TRAIN],
                                  batch_size=self.kwargs['train_batch'],
                                  **self.cfg['dataloader'],
                                  collate_fn=self.dataset[KEY_TRAIN].collate_fn)
        dev_loader = DataLoader(dataset=self.dataset[KEY_DEV],
                                batch_size=self.kwargs['dev_batch'],
                                **self.cfg['dataloader'],
                                collate_fn=self.dataset[KEY_DEV].collate_fn)

        for epoch in range(self.epoch_start, self.epoch_num):
            step = True if (epoch + 1) % self.update_every == 0 else False
            self.train_once(epoch, train_loader, self.model, self.device, step)
            if self.validate_after < epoch and (epoch + 1) % self.validate_every == 0:
                self.eval(epoch, dev_loader, self.model, self.dev_device)
            if self.scheduler:
                self.scheduler.step(epoch=epoch)
            if self.save_strategy == SAVE_STRATEGY_ALL:
                self.checkpoint(epoch)
            if self.early_stop and self.stop_counter > EARLY_STOP_THRESHOLD:
                break  # todo 检查机制待完善

        self.writer.close()
        return

    def train_once(self,
                   epoch: int,
                   loader: DataLoader,
                   model: Model,
                   device: torch.device,
                   step: bool = True):
        """
        """
        time_epoch_start = time.time()
        loss_epoch = 0
        model.to(device)
        model.train_mode()
        if self.log_batch:
            tqdm_desc = f"Train epoch {epoch}"
        else:
            tqdm_desc = f"[{sys_info()}] Train epoch {epoch}"

        for batch_i, batch in Tqdm(enumerate(loader), total=len(loader), desc=tqdm_desc):
            to_device(batch, device)
            loss = model(**batch)['loss']
            loss_epoch += loss.item()
            (loss / self.update_every).backward()  # gradient accumulation
            if step:
                if self.clip_grad:
                    clip_grad_func(model.parameters(), **self.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad()

            if batch_i % self.log_interval == 0:
                if self.log_batch:
                    Tqdm.write(f"[{sys_info()}] {batch_i}/{len(loader)} : "
                               f"Loss= {loss.item():.4f}")
                n_example = (epoch * len(loader) + batch_i) * loader.batch_size
                self.writer.add_scalar('Train/loss', loss.item(), n_example)

        self.time_epoch = time.time() - time_epoch_start
        loss_epoch /= len(loader)
        lr = self.scheduler.get_lr()
        if isinstance(lr, List) and len(lr) == 1:
            self.writer.add_scalar('Train/learning_rate', lr[0], epoch)
        else:
            raise NotImplementedError("我还没想咋写")
        self.writer.add_scalar('Train/epoch_loss', loss_epoch, epoch)
        self.writer.flush()
        Tqdm.write(f"===> Epoch {epoch} compete, avg loss {loss_epoch:.4f}, "
                   f"time {sec_to_time(self.time_epoch)}, "
                   f"remaining {sec_to_time(self.time_left())}")
        return

    def eval(self,
             epoch: int,
             loader: DataLoader,
             model: Model,
             device: torch.device):
        time_eval_start = time.time()
        loss_epoch = 0
        model.to(device)
        model.eval_mode()
        if self.log_batch:
            tqdm_desc = f"Eval at no.{epoch}"
        else:
            tqdm_desc = f"[{sys_info()}] Eval at no.{epoch}"

        for batch_i, batch in Tqdm(enumerate(loader), desc=tqdm_desc, total=len(loader)):
            to_device(batch, device)
            with torch.no_grad():
                output = model(**batch)
            loss, metric = output['loss'], output['metric']
            loss_epoch += loss.item()

            if batch_i % self.log_interval == 0:
                if self.log_batch:
                    Tqdm.write(f"\r[{sys_info()}] {batch_i}/{len(loader)} : "
                               f"Loss: {loss.item():.4f}")
                n_example = (epoch * len(loader) + batch_i) * loader.batch_size
                self.writer.add_scalar('Dev/loss', loss.item(), n_example)

        self.time_eval = time.time() - time_eval_start
        loss_epoch /= len(loader)
        metric = model.get_metrics(reset=True)
        metric['epoch_loss'] = loss_epoch
        self.add_scalars('Dev', metric, epoch)
        self.writer.flush()

        Tqdm.write(f"===> Eval compete, time {sec_to_time(self.time_eval)}, "
                   f"remaining {sec_to_time(self.time_left())}, "
                   f"{', '.join([f'{k}: {v:.4f}' for k, v in metric.items()])}")

        if self.save_after > epoch:
            return

        if self.save_strategy != SAVE_STRATEGY_BEST:
            return

        if self.best_metric:
            if self.model.is_best(metric, self.best_metric):
                self.best_metric = metric
                self.checkpoint(epoch, 'best')
                self.stop_counter = 0
            elif loss_epoch > self.best_metric['epoch_loss']:
                self.stop_counter += 1
        else:
            self.best_metric = metric
            self.checkpoint(epoch, 'best')
            self.stop_counter = 0
        return

    def test(self):
        return

    def checkpoint(self, epoch: int, comment: str = ''):
        self.cfg = self.cfg.reload()

        for key in ('epoch_num', 'update_every', 'validate_every',
                    'validate_after', 'save_after', 'log_interval'):
            if key in self.cfg['trainer']:
                self.__setattr__(key, self.cfg['trainer'][key])

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }

        path = os.path.normpath(f"{self.save_dir}/{self.prefix}_{epoch}_{comment}.bac")
        torch.save(checkpoint, path)

        self.cfg['trainer']['pre_train_path'] = path
        self.cfg['trainer']['epoch_start'] = epoch + 1
        self.cfg.save()

        Tqdm.write(f"=======> Checkpoint saved to {path}")

    def load(self):
        checkpoint = torch.load(self.kwargs['pre_train_path'])
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return self

    def add_scalars(self, main_tag: str, value_dict: Dict[str, Any],
                    global_step: int, key_prefix=None):
        for key, value in value_dict.items():
            key = f'{key_prefix}_{key}' if key_prefix else key
            self.writer.add_scalar(f"{main_tag}/{key}", value, global_step)
