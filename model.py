#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

from pathlib import Path

import torch
import torchvision.models as M

from utils import *


TORCHVISION_MODELS = M.list_models()

def get_model(name, ckpt_fp=None) -> Module:
  if not hasattr(M, name): raise ValueError(f'[get_model] unknown model {name}')
  model: Module = getattr(M, name)(pretrained=ckpt_fp is None)
  if ckpt_fp: model.load_state_dict(torch.load(ckpt_fp))
  return model


# repo: https://github.com/huyvnphan/PyTorch_CIFAR10
CKPT_PATH = Path('repo/PyTorch_CIFAR10/cifar10_models/state_dicts')
PYTORCH_CIFAR10_MODELS = sorted([fp.stem for fp in CKPT_PATH.iterdir()]) if CKPT_PATH.is_dir() else []

def get_model_pytorch_cifar10(name) -> Module:
  pass
