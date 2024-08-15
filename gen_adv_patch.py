#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/06/16 

from pathlib import Path
import os
from time import time
from argparse import ArgumentParser
from typing import *

import numpy as np
import torch
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AdversarialPatchPyTorch
from tqdm import tqdm

from data import CIFAR10Data
from module import all_classifiers

from data import *
from model import *
from utils import *

OUT_PATH = Path('out') ; OUT_PATH.mkdir(exist_ok=True)


def get_atk(args, model:Module) -> AdversarialPatchPyTorch:
  clf = PyTorchClassifier(
    model=model,
    loss=CrossEntropyLoss(),
    input_shape=(32, 32, 3),
    nb_classes=10,
    optimizer=None,
    clip_values=(0.0, 1.0),
    preprocessing_defences=None,
    postprocessing_defences=None,
    preprocessing=DA(),   # normalize
    device_type="gpu",
  )
  
  if args.ap_pgd:
    optimizer     = "pgd"
    learning_rate = 1/255
  else:
    optimizer     = "Adam"
    learning_rate = 5.0

  # NOTE: when got error "AdversarialPatchPyTorch requires torch>=1.7.0"
  # manually comment the version checking lines in the `art` library
  ap = AdversarialPatchPyTorch(
    estimator=clf,
    rotation_max=0.0,
    scale_min=1.0,
    scale_max=1.0,
    distortion_scale_max=0.0,
    optimizer=optimizer,
    learning_rate=learning_rate,
    max_iter=args.ap_iter,
    batch_size=args.batch_size,
    patch_shape=(3, args.ap_size, args.ap_size),
    patch_location=None,
    patch_type=args.ap_shape,
    targeted=False,
    summary_writer=False,
    verbose=True,
  )

  return ap

@torch.enable_grad()
def run(args, model:Module, dataloader:DataLoader, atk:AdversarialPatchPyTorch):
  adv_patches = []

  model.eval()
  for X, Y in tqdm(dataloader):
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    P_np, M_np = atk.generate(X_np, Y_np)
    adv_patches.append(P_np)

  adv_patches = np.concatenate(adv_patches, axis=0)

  out_dp = OUT_PATH / args.model ; out_dp.mkdir(exist_ok=True)
  breakpoint()


def go(args):
  model = all_classifiers[args.model]()
  state_dict = os.path.join("cifar10_models", "state_dicts", args.model + ".pt")
  model.load_state_dict(torch.load(state_dict))
  model = model.eval().to(device)

  data = CIFAR10Data(args)
  trainloader = data.train_dataloader()

  atk = get_atk(args, model)

  t = time()
  run(args, model, trainloader, atk)
  print(f'Done in ({time() - t:.3f} s)')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model',      default='resnet50', choices=PYTORCH_CIFAR10_MODELS, help='model to attack')
  parser.add_argument('-B', '--batch_size', default=1, type=int, help='batch size')
  parser.add_argument("--data_dir", type=str, default='data')
  parser.add_argument("--num_workers", type=int, default=0)
  parser.add_argument('--ap_size',  default=5,  type=int, help='attack patch size, typically 5x5 patch over 32x32 image')
  parser.add_argument('--ap_bs',    default=24, type=int, help='attack batch size')
  parser.add_argument('--ap_shape', default='square', choices=['circle', 'square'], help='patch shape')
  parser.add_argument('--ap_rot',   default=22.5, type=float, help='max patch rotation angle')
  parser.add_argument('--ap_pgd',   action='store_true',      help='optim method, pgd or simple Adam')
  parser.add_argument('--ap_iter',  default=100,  type=int,   help='optim iter')
  args = parser.parse_args()

  go(args)
