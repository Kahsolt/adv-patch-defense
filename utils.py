#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

from time import time
from typing import *
import warnings ; warnings.filterwarnings("ignore", category=UserWarning)

import random
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

seed = 114514
npimg = np.ndarray

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random      .seed(seed)
np.random   .seed(seed)
torch.manual_seed(seed)
if device == 'cuda':
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = True
  torch.cuda.manual_seed(seed)


def perf_count(fn):
  def wrapper(*args, **kwargs):
    if device == 'cuda':
      torch.cuda.reset_peak_memory_stats()
      torch.cuda.ipc_collect()

    t = time()
    r = fn(*args, **kwargs)
    print(f'>> done in {time() - t:.3f}s')

    if device == 'cuda':
      alloc = torch.cuda.max_memory_allocated() // 2**20
      resrv = torch.cuda.max_memory_reserved()  // 2**20
      print(f'[vram] alloc: {alloc} MB, resrv: {resrv} MB')

    return r
  return wrapper
