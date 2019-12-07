from typing import Callable, Dict, Any
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from tjunlp.common.checks import ConfigurationError
from tjunlp.common.config import Config
from tjunlp.common.tqdm import Tqdm
from tjunlp.common.util import sys_info, sec_to_time, merge_dicts
from tjunlp.core.dataset import DataSet


def to_device(data, device: torch.device):
    # def to(data: torch.Tensor, device: torch.device):
    #     if device == data.device:
    #         return
    #     elif device.type == 'cpu':
    #         data.cpu()
    #     else:
    #         data.cuda(device)
    #     return data

    if isinstance(data, Dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, torch.Tensor):
        data = data.to(device)
    else:
        pass
    return data


class Trainer(object):
    """

    """

    def __init__(self,
                 cfg: Config,
                 dataset: Dict[str, DataSet],
                 model: nn.Module,
                 optimizer: Optimizer,
                 device: str,
                 n_epochs: int,
                 dev_on_cpu: bool = True,
                 log_dir: str = '~/tb_log/',
                 prefix: str = 'unknown',
                 **kwargs):
        self.cfg = cfg
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.kwargs = kwargs
        if device == 'cuda' and not torch.cuda.is_available():
            raise ConfigurationError("No GPU found, please run at CPU!")
        self.device = torch.device(device)
        self.n_epochs = n_epochs
        # self.update_every = update_every
        # self.validate_every = validate_every
        # self.save_after = save_after
        # self.log_interval = log_interval  # log every X batches
        self.dev_device = torch.device('cpu') if dev_on_cpu else self.device
        self.time_epoch = 0
        self.time_eval = 0
        self.best_metric = None

        if not os.path.exists(os.path.abspath(log_dir)):
            os.mkdir(log_dir)
        path = f"{log_dir}/{prefix + str(datetime.now())[:16].replace(' ', '_')}"
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        os.mkdir(path)
        self.writer = SummaryWriter(log_dir=path)

        return

    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    def time_left(self):
        if self.time_eval == 0:
            self.time_eval = self.time_epoch
        time_left = (self.time_epoch + self.time_eval / self.time_eval) * self.n_epochs
        return time_left

    def train(self):
        train_loader = DataLoader(dataset=self.dataset['train'],
                                  batch_size=self.kwargs['train_batch'],
                                  **self.cfg['dataloader'],
                                  collate_fn=self.dataset['train'].collate_fn)
        dev_loader = DataLoader(dataset=self.dataset['dev'],
                                batch_size=self.kwargs['dev_batch'],
                                **self.cfg['dataloader'],
                                collate_fn=self.dataset['train'].collate_fn)

        for e in range(self.n_epochs):
            if (e + 1) % self.kwargs['update_every'] == 0:
                step = True
            else:
                step = False
            self.train_once(e, self.kwargs['log_interval'], train_loader,
                            self.model.forward, self.device, step)
            if (e + 1) % self.kwargs['validate_every'] == 0:
                self.eval(e, self.kwargs['log_interval'], dev_loader,
                          self.model.forward, self.dev_device)

        self.writer.close()
        return

    def train_once(self,
                   epoch: int,
                   log_interval: int,
                   loader: DataLoader,
                   model_forward: Callable,
                   device: torch.device,
                   step: bool = True):
        """
        """
        time_epoch_start = time.time()
        loss_epoch = 0
        self.model.to(device)
        self.model.train()

        for batch_i, batch in Tqdm(enumerate(loader), desc=f'Train epoch {epoch}'):
            to_device(batch, device)
            loss = model_forward(**batch)['loss']
            loss_epoch += loss.item()
            # TODO:  梯度剪辑
            loss.backward()
            if step:
                self.optimizer.step()
                self.optimizer.zero_grad()

            Tqdm.write(f"[{sys_info()}] {batch_i}/{len(loader)} : "
                       f"Loss= {loss.item():.4f}", nolock=True)
            if batch_i % log_interval == 0:
                n_example = (epoch * len(loader) + batch_i) * loader.batch_size
                self.writer.add_scalar('Train/loss', loss.item(), n_example)

        self.time_epoch = time.time() - time_epoch_start
        loss_epoch /= len(loader)
        self.writer.add_scalar('Train/learning_rate', self.lr(), epoch)
        self.writer.add_scalar('Train/epoch_loss', loss_epoch, epoch)
        self.writer.flush()
        Tqdm.write(f"Epoch {epoch} compete, avg loss {loss_epoch:.4f}, "
                   f"time {sec_to_time(self.time_epoch)}, "
                   f"remaining {sec_to_time(self.time_left())}")
        return

    def eval(self,
             epoch: int,
             log_interval: int,
             loader: DataLoader,
             model_forward: Callable,
             device: torch.device):
        time_eval_start = time.time()
        metric_list = list()
        loss_epoch = 0
        self.model.to(device)
        self.model.eval()

        for batch_i, batch in Tqdm(enumerate(loader), desc=f'Eval', total=len(loader)):
            to_device(batch, device)
            with torch.no_grad():
                output = model_forward(**batch)
            loss, metric = output['loss'], output['metric']
            loss_epoch += loss.item()
            metric_list.append(metric)

            Tqdm.write(f"\r[{sys_info()}] {batch_i}/{len(loader)} : "
                       f"Loss: {loss.item():.4f}")
            if batch_i % log_interval == 0:
                n_example = (epoch * len(loader) + batch_i) * loader.batch_size
                metric['loss'] = loss.item()
                self.add_scalars('Dev', metric, n_example)

        self.time_eval = time.time() - time_eval_start
        loss_epoch /= len(loader)
        metric = merge_dicts(metric_list, 'epoch_')
        metric['epoch_loss'] = loss_epoch
        self.add_scalars('Dev', metric, epoch)
        self.writer.flush()
        Tqdm.write(f"Epoch {epoch} compete, avg loss {loss_epoch:.4f}, "
                   f"time {sec_to_time(self.time_eval)}, "
                   f"remaining {sec_to_time(self.time_left())}")

        if self.kwargs['save_after'] > epoch:
            return

        if self.best_metric:
            if self.model.best(metric, self.best_metric):
                self.checkpoint(epoch, 'best')
        else:
            self.checkpoint(epoch, 'best')

        return

    def test(self):
        return

    def checkpoint(self, epoch: int, comment: str = ''):
        self.cfg = self.cfg.reload()
        self.n_epochs = self.cfg['trainer']['n_epochs']  # may change

        path = f"{self.cfg['trainer']['prefix']}_{epoch}_{comment}.pth"
        torch.save(self.model.state_dict(), path)

        self.cfg['trainer']['pre_train_path'] = path
        self.cfg['trainer']['pre_trained'] = True
        self.cfg.save()

        # with open(f"{opt['save_dir']}/optim.pkl", 'wb') as f:
        #     pickle.dump(optimizer, f)

        Tqdm.write(f"Checkpoint saved to {path}")

    def add_scalars(self, main_tag: str, value_dict: Dict[str, Any], global_step: int):
        for key, value in value_dict.items():
            self.writer.add_scalar(f'{main_tag}/{key}', value, global_step)
