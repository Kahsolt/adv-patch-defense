#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

from pathlib import Path

import torch
import torchvision.models as M

from utils import *


TORCHVISION_MODELS = M.list_models()

def get_model(name:str, ckpt_fp:str=None) -> Module:
  if not hasattr(M, name): raise ValueError(f'[get_model] unknown model {name}')
  model: Module = getattr(M, name)(pretrained=ckpt_fp is None)
  if ckpt_fp: model.load_state_dict(torch.load(ckpt_fp))
  return model


# repo: https://github.com/huyvnphan/PyTorch_CIFAR10
PYTORCH_CIFAR10_PATH = Path('repo') / 'PyTorch_CIFAR10'
PYTORCH_CIFAR10_CKPT_PATH = PYTORCH_CIFAR10_PATH / 'cifar10_models' / 'state_dicts'
PYTORCH_CIFAR10_MODELS = sorted([fp.stem for fp in PYTORCH_CIFAR10_CKPT_PATH.iterdir()]) if PYTORCH_CIFAR10_CKPT_PATH.is_dir() else []

def get_model_pytorch_cifar10(name:str) -> Module:
  assert PYTORCH_CIFAR10_PATH.is_dir(), 'huyvnphan/PyTorch_CIFAR10 not found under repo folder, forgot to run init_repos.cmd?'
  import sys; sys.path.append(PYTORCH_CIFAR10_PATH)
  from module import all_classifiers
  model: Module = all_classifiers[name]()
  fp = PYTORCH_CIFAR10_CKPT_PATH / (name + ".pt")
  assert fp.is_file(), f'pretrained weights file not found: {fp}'
  model.load_state_dict(torch.load(fp))
  return model
