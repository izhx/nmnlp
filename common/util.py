import os
import re
import logging
import subprocess
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Union

import torch
# import psutil

logger = logging.getLogger(__name__)

BYTE_GB = 1073741824  # 1024*1024*1024
MG_GB = 1024


def now():
    return str(datetime.now())[:-7]


def output(*args):
    message = ''.join([str(arg) for arg in args])
    print(f"[{now()}] {message}")


def field_match(pattern: str, namespace: str):
    """
    Matches a namespace pattern against a namespace string.  For example, ``*tags`` matches
    ``passage_tags`` and ``question_tags`` and ``tokens`` matches ``tokens`` but not
    ``stemmed_tokens``.
    """
    if pattern[0] == '*' and namespace.endswith(pattern[1:]):
        return True
    elif pattern == namespace:
        return True
    return False


def gpu_memory_mb() -> Dict[int, List[int]]:
    """
    Get the current GPU memory usage.
    Based on https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    Returns
    -------
    ``Dict[int, int]``
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
        Returns an empty ``dict`` if GPUs are not available.
    """
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,nounits,noheader"], encoding="utf-8",
        )
        info = [x.split(',') for x in result.strip().split("\n")]
        return {gpu: [int(mem[0]), int(mem[1])] for gpu, mem in enumerate(info)}
    except FileNotFoundError:
        # `nvidia-smi` doesn't exist, assume that means no GPU.
        return {}
    except:  # noqa
        # Catch *all* exceptions, because this memory check is a nice-to-have
        # and we'd never want a training run to fail because of it.
        logger.exception("unable to check gpu_memory_mb(), continuing")
        return {}


# def sys_info() -> str:
#     gpu_info = [f"GPU-{k}: {(v[0] / MG_GB):.2f}/{(v[1] / MG_GB):.2f}GB"
#                 for k, v in gpu_memory_mb().items()]
#     mem = psutil.virtual_memory()
#     return f"CPU: {psutil.cpu_percent()}%, " \
#            f"Mem: {(mem.used / BYTE_GB):.1f}/{(mem.total / BYTE_GB):.1f}GB, " \
#            f"{', '.join(gpu_info)}"


def sec_to_time(seconds) -> str:
    if seconds < 60:
        return f"{str(int(seconds)).rjust(2)}s"
    m, s = divmod(seconds, 60)
    time = f"{str(int(s)).rjust(2)}s"
    if m < 60:
        return f"{str(int(m)).rjust(2)}m " + time
    h, m = divmod(m, 60)
    time = f"{str(int(m)).rjust(2)}m " + time
    if h < 24:
        return f"{str(int(h)).rjust(2)}h " + time
    d, h = divmod(h, 24)
    return f"{str(int(d)).rjust(2)}d {str(int(h)).rjust(2)}h " + time


def merge_dicts(dicts: Union[List[Dict], Dict[str, Dict]], key_prefix='',
                avg=False) -> Dict:
    result: Dict[str, int] = defaultdict(lambda: 0)
    if isinstance(dicts, Dict):
        dicts = dicts.values()

    for d in dicts:
        for k, v in d.items():
            result[key_prefix + k] += v

    if avg:
        for k in result:
            result[k] /= len(dicts)

    return result


def set_visible_devices(cuda_ids: str) -> Union[torch.device, List[torch.device]]:
    """ 1 2 3  or 1,3,2 or 3, 2, 1 必须有数字以外的分隔符，变换顺序可以映射gpu id"""
    cuda_ids = re.findall(r"\d+", cuda_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(cuda_ids)
    if len(cuda_ids) == 0:
        return torch.device('cpu')
    if len(cuda_ids) > 1:
        return [torch.device(f'cuda:{i}') for i in range(len(cuda_ids))]
    return torch.device('cuda:0')


def to_device(data, device: torch.device):
    if torch.is_tensor(data):
        data = data.to(device)
    elif isinstance(data, dict):
        data = {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        data = [to_device(i, device) for i in data]

    return data
