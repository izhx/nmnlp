"""
My trainer.
"""

from typing import Dict, Any, List, Union, Callable
import os
import time
import copy
import warnings
from argparse import Namespace
from collections import OrderedDict

import torch
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim.optimizer import Optimizer  # pylint: disable=no-name-in-module
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard.writer import SummaryWriter

from ..common.config import load_yaml, save_yaml
from ..common.util import sec_to_time, merge_dicts, output, to_device
from ..data import index_dataset
from .dataset import DataSet
from .optim import get_lrs
from .vocabulary import Vocabulary


EARLY_STOP_THRESHOLD = 10
LOG_INTERVAL_DENOMINATOR = 100

SAVE_STRATEGY_NO = 'no'
SAVE_STRATEGY_ALL = 'all'
SAVE_STRATEGY_BEST = 'best'
SAVE_STRATEGY_SKIP = 'skip'  # every 2

DEFAULT_MODEL_SAVE_DIR = './exp/'
DEFAULT_LOG_DIR = './tblog/'
DEFAULT_PREFIX = 'model_'

DEVICE_CPU = 'cpu'
DEVICE_CUDA = 'cuda'

CALLBACKS = ('before_epoch_start', 'after_collate_batch', 'after_batch_forward',
             'before_next_batch', 'after_epoch_end')


def format_metric(metric: Dict) -> str:
    info = reversed([f"{k}: {v:.4f}" for k, v in metric.items()])
    return ', '.join(info)


def clip_grad_func(parameters, method: str, **kwargs):
    if method == 'norm':
        clip_grad_norm_(parameters, **kwargs)
    elif method == 'value':
        clip_grad_value_(parameters, **kwargs)
    else:
        raise ValueError("Wrong gradient clip type!")


def forever_yield_data(dataset, batch_size, shuffle, sampler):
    loader = DataLoader(
        dataset, batch_size, shuffle, sampler, collate_fn=dataset.collate_fn)
    while True:
        for batch in loader:
            yield batch
    warnings.warn("Unexpected exit.")


class Trainer(object):
    """
    有监督学习的trainer
    """

    def __init__(self,
                 cfg: Dict[str, Any],
                 dataset: Union[Namespace, Dict[str, DataSet]],
                 vocabulary: Vocabulary,
                 model: torch.nn.Module,
                 optimizer: Optimizer,
                 sampler: Sampler = None,  # train data sampler
                 scheduler: Any = None,  # _LRScheduler is protected
                 logger: Any = None,
                 device: str = DEVICE_CPU,
                 clip_grad: Dict = None,
                 batch_size: int = 1,
                 early_stop: bool = True,
                 epoch_num: int = 100,
                 epoch_start: int = 0,
                 test_every: int = 0,
                 update_every: int = 1,
                 validate_every: int = 1,
                 validate_after: int = 0,
                 save_after: int = 30,
                 save_dir: str = DEFAULT_MODEL_SAVE_DIR,
                 save_not_only_model: bool = False,
                 save_strategy: str = SAVE_STRATEGY_BEST,
                 tensorboard: bool = False,
                 log_batch: bool = False,
                 log_dir: str = DEFAULT_LOG_DIR,
                 log_interval: int = 100,
                 prefix: str = DEFAULT_PREFIX,
                 pre_train_path: str = None,
                 **kwargs):
        self.cfg = cfg
        self.dataset = dataset if isinstance(dataset, Namespace) else Namespace(**dataset)
        self.model = model
        self.optimizer = optimizer
        self.sampler = sampler
        self.scheduler = scheduler
        self.clip_grad = clip_grad
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.epoch_num = epoch_num
        self.epoch_start = epoch_start if pre_train_path else 0
        self.test_every = test_every  # default 0, 不测试
        self.update_every = update_every  # 梯度累积的步数 i.e. accumulation_steps
        self.validate_every = validate_every
        if validate_every < update_every or validate_every % update_every != 0:
            raise ValueError("You can't validate and save before step() !")
        self.save_after = save_after
        self.save_dir = save_dir
        self.save_not_only_model = save_not_only_model
        self.save_strategy = save_strategy
        self.prefix = prefix
        self.pre_train_path = pre_train_path
        self.device = torch.device(device)
        self.time_epoch, self.time_eval = 0, 0

        self.callbacks = Namespace(**{
            k: model.__dict__[k] if k in model.__dict__ else lambda *_: _
            for k in CALLBACKS})

        index_dataset(dataset, vocabulary)

    def time_left(self, epoch):
        self.time_eval = self.time_epoch / 7 if self.time_eval == 0 else self.time_eval
        time_left = (self.time_epoch + self.time_eval / self.validate_every
                     ) * (self.epoch_num - epoch)
        return time_left

    def get_loader(self, dataset: Union[DataSet, Dict], batch_size: int = 0, shuffle: bool = False,
                   sampler: Sampler = None):
        batch_size = self.batch_size if batch_size == 0 else batch_size
        if isinstance(dataset, dict):
            warnings.warn("got dict but not DataSet, will return iters.")
            iters = OrderedDict({k: forever_yield_data(
                v, batch_size, shuffle, sampler) for k, v in dataset.items()})
            return iters
        return DataLoader(dataset, batch_size, shuffle, sampler,
                          collate_fn=dataset.collate_fn)

    def train(self) -> bool:
        train_loader = self.get_loader(
            self.dataset.train, shuffle=self.sampler is None, sampler=self.sampler)
        output(f"batch num (per epoch): {len(train_loader)}")
        run_flag = True  # 是否继续训练
        epoch, best_epoch, stop_counter = self.epoch_start, 0, 0
        output(f"Training started at epoch {epoch} ...")
        time_start = time.time()

        while epoch <= self.epoch_num and run_flag:
            step = bool((epoch + 1) % self.update_every == 0)  # 是否反向传播
            self._train_once(epoch, train_loader, step)

            if (epoch + 1) % self.validate_every == 0:
                metric = self._eval_once(epoch, self.dataset.dev)
                if self.model.metric.is_best(metric):
                    best_epoch, stop_counter = epoch, 0
                    if self.save_strategy == SAVE_STRATEGY_BEST:
                        self.checkpoint(epoch, comment='best')
                else:
                    stop_counter += 1

            if self.test_every > 0 and (epoch + 1) % self.test_every == 0:
                self.test(self.dataset.test, self.batch_size)

            self.reload_cfg()
            if self.save_strategy != SAVE_STRATEGY_NO:
                if self.save_strategy == SAVE_STRATEGY_ALL and epoch > self.save_after:
                    self.checkpoint(epoch)
            if self.early_stop and stop_counter > EARLY_STOP_THRESHOLD:
                run_flag = False  # 检查机制待完善
                output('Early stoped!')
            epoch += 1

        time_train = time.time() - time_start
        output(f'training compete, time: {sec_to_time(time_train)} .')
        output(f"Best epoch: {best_epoch}, "
               f"{format_metric(self.model.metric.best)}")
        if self.writer:
            self.writer.close()
        return run_flag  # 若早停，返回false

    def _train_once(self, epoch: int, loader, step: bool):
        self.model.to(self.device)
        self.model.train()
        self.callbacks.before_epoch_start(locals())

        time_start = time.time()
        losses = self.train_func(loader, epoch, step)
        self.time_epoch = time.time() - time_start

        loss_epoch = losses.mean().item()
        self.callbacks.after_epoch_end(locals())

        if self.writer:
            scalars = dict(get_lrs(self.optimizer))
            scalars['epoch_loss'] = loss_epoch
            scalars['loss_variance'] = losses.var()
            self.add_scalars('Train', scalars, epoch)

        output(f"Epoch {epoch} compete, epoch_loss: {loss_epoch:.4f}, "
               f"time: {sec_to_time(self.time_epoch)}, remaining: "
               f"{sec_to_time(self.time_left(epoch))}.")

    def train_func(self, loader, epoch, step) -> torch.Tensor:
        """ the training procedure of one epoch.
        """
        losses = torch.zeros(len(loader), device=self.device)
        log_interval = max(int(len(loader) / LOG_INTERVAL_DENOMINATOR), 1)

        for i, (input_dict, batch) in enumerate(loader):

            input_dict, *_ = self.callbacks.after_collate_batch(input_dict, batch, locals())
            input_dict = to_device(input_dict, self.device)

            output_dict = self.model(**input_dict)

            output_dict, *_ = self.callbacks.after_batch_forward(output_dict, locals())

            loss = output_dict['loss']
            losses[i] = loss.item()
            if self.update_every == 1:
                loss.backward()
            else:
                (loss / self.update_every).backward()  # gradient accumulation
            if step:
                if self.clip_grad:
                    clip_grad_func(self.model.parameters(), **self.clip_grad)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()

            if i % log_interval == 0 and self.writer:
                n_example = (epoch * len(loader) + i) * loader.batch_size
                self.writer.add_scalar('Train/loss', loss.item(), n_example)

            self.callbacks.before_next_batch(locals())

        return losses

    def _eval_once(self, epoch: int, dataset: Union[DataSet, List[DataSet], Dict[str, DataSet]]):
        def eval_one(one_set, name):
            return self.process_one(one_set, name, self.dev_device, self.batch_size, epoch)

        self.model.eval()
        time_eval_start = time.time()

        with torch.no_grad():
            _, metric, losses = self._process_many(dataset, eval_one, epoch)

        self.time_eval = time.time() - time_eval_start
        metric['loss_variance'] = losses.var().item()
        metric['epoch_loss'] = losses.mean().item()
        if self.writer:
            self.add_scalars('Dev', metric, epoch)
            self.writer.flush()
        output(f"Eval compete, {format_metric(metric)}")

        return metric

    def test(self, dataset: Union[DataSet, List[DataSet], Dict[str, DataSet]],
             batch_size: int = 0, device: torch.device = None):
        device = self.device if device is None else device

        def test_one(one_set, name):
            return self.process_one(one_set, name, device, batch_size)

        self.model.train(False)  # equal to `self.model.eval()`
        with torch.no_grad():
            counters, *_ = self._process_many(dataset, test_one)

        return counters

    def _process_many(self, dataset: Union[DataSet, List[DataSet], Dict[str, DataSet]],
                      func: Callable, epoch=None):
        if isinstance(dataset, DataSet):
            return func(dataset, '')
        if len(dataset) < 1:
            raise ValueError('Dataset is empty!')
        if isinstance(dataset, Dict):
            iterator = dataset.items()
        elif isinstance(dataset, List):
            iterator = enumerate(dataset)
        else:
            raise ValueError('dataset type not support!')

        counters, losses = list(), list()
        for name, one_set in iterator:
            counter, _, loss = func(one_set, name)
            counters.append(counter)
            losses.append(loss)

        metric = self.model.metric.get_metric(counter=merge_dicts(counters))
        if epoch is None:
            output(f"All compete, {format_metric(metric)}")
        return counters, metric, torch.cat(losses)

    def process_one(self, one_set, name, device, batch_size, epoch=None):
        """ epoch is None means test stage.
        """
        loader = self.get_loader(one_set, batch_size)
        len_loader = len(loader)
        losses = torch.zeros(len_loader, device=device)

        for i, (input_dict, batch) in enumerate(loader):
            input_dict, *_ = self.callbacks.after_collate_batch(input_dict, batch, locals())
            input_dict = to_device(input_dict, device)

            output_dict = self.model(**input_dict)

            output_dict, *_ = self.callbacks.after_batch_forward(output_dict, locals())

            losses[i] = output_dict['loss'].item()

            self.callbacks.before_next_batch(locals())

        metric_counter = copy.deepcopy(self.model.metric.counter)
        metric = self.model.metric.get_metric(reset=True)
        if epoch is not None and self.writer is not None:
            metric['loss'] = losses.mean()
            self.add_scalars('Very_Detail', metric, epoch, name)
            self.writer.flush()
        elif epoch is None:
            output(f"Test {name} compete, {format_metric(metric)}")
        return metric_counter, metric, losses

    def checkpoint(self, epoch: int, comment: str = None):
        """
        if comment is not none, save path won't use epoch!
        """
        self.reload_cfg()
        post_fix = 'bac' if self.save_not_only_model else 'pth'

        if comment is None:
            path = os.path.normpath(f"{self.save_dir}/{self.prefix}_{epoch}.{post_fix}")
        else:
            path = os.path.normpath(f"{self.save_dir}/{self.prefix}_{comment}.{post_fix}")

        if self.save_not_only_model:
            self.cfg['trainer']['pre_train_path'] = path
            self.cfg['trainer']['epoch_start'] = epoch + 1

            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'cfg': self.cfg,
                'epoch': epoch
            }
            if self.log_dir:
                checkpoint['log_dir'] = self.log_dir
            if self.scheduler:
                checkpoint['scheduler'] = self.scheduler.state_dict()
        else:
            checkpoint = self.model.state_dict()

        torch.save(checkpoint, path)
        save_yaml(self.cfg)
        self.pre_train_path = path
        output(f"Checkpoint saved at <{path}>")

    def load(self):
        checkpoint = torch.load(self.pre_train_path, map_location=self.device)

        if self.pre_train_path.endswith('bac'):
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'log_dir' in checkpoint and self.log_dir is not None:  # 如果存档有，并且初始化时没要求不用tb
                self.log_dir = checkpoint['log_dir']
                self.writer = SummaryWriter(log_dir=self.log_dir)
            output(f"Loaded checkpoint at epoch {checkpoint['epoch']} "
                   f"from <{self.pre_train_path}>")
        else:
            self.model.load_state_dict(checkpoint)
            output(f"Loaded model checkpoint from <{self.pre_train_path}>")

        return self

    def reload_cfg(self):
        self.cfg = load_yaml(self.cfg['path'])

        for key in ('epoch_num', 'validate_every', 'save_after',
                    'save_strategy', 'log_batch', 'log_interval'):
            if key in self.cfg['trainer']:
                self.__setattr__(key, self.cfg['trainer'][key])
        return

    # def before_epoch_start(self, local_dict):
    #     """
    #     """
    #     pass

    # def after_collate_batch(self, input_dict, batch, local_dict):
    #     """
    #     在dataloader获取到一个batch数据后调用。
    #     """
    #     pass

    # def after_batch_forward(self, input_dict, output_dict, local_dict):
    #     """
    #     模型forword之后，反向传播等之前。
    #     """
    #     pass

    # def before_next_batch(self, local_dict):
    #     """
    #     反向传播、log之后，加载下一batch前。
    #     """
    #     pass

    # def after_epoch_end(self, local_dict):
    #     """
    #     """
    #     pass
