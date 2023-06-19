#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import sys
from pathlib import Path

from time import time
from typing import *
import warnings ; warnings.filterwarnings("ignore", category=UserWarning)

import random
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

REPO_PATH = Path('repo')
if 'repo':
  PYTORCH_CIFAR10_PATH = REPO_PATH / 'PyTorch_CIFAR10'
  
  IP_PATH = REPO_PATH / 'ImageNet-Patch'
  IP_FILE = IP_PATH / 'assets' / 'imagenet_patch.gz'

  SAC_PATH = REPO_PATH / 'SegmentAndComplete'
  SAC_CKPT = SAC_PATH / 'ckpts' / 'coco_at.pth'

  MAE_PATH = REPO_PATH / 'mae'
  MAE_CKPT = MAE_PATH / 'models' / 'mae_visualize_vit_large_ganloss.pth'
  MAE_PATCH_SIZE = 16

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
