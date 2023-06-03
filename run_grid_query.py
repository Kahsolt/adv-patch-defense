#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/24 

import gzip
import lzma
import pickle as pkl
from pathlib import Path
from time import time
from argparse import ArgumentParser

from torch.nn import Module
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

if 'repos':
  import sys
  sys.path.append('repo/ImageNet-Patch')
  from transforms.apply_patch import ApplyPatch
  PACTCH_FILE = "repo/ImageNet-Patch/assets/imagenet_patch.gz"

from utils import *

BASE_PATH = Path(__file__).parent.absolute()
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)


@torch.inference_mode()
def run(args, model:Module, X:Tensor, Y:Tensor, P:Tensor):
  B, C, H, W = X.shape ; assert B == 1
  pB, pC, pH, pW = P.shape ; assert C == pC
  shape = qH, qW = [H - pH + 1, W - pW + 1]
  logits = np.zeros(shape, dtype=np.float32)
  probs  = np.zeros(shape, dtype=np.float32)
  losses = np.zeros(shape, dtype=np.float32)
  preds  = np.zeros(shape, dtype=np.int32)

  total = qH * qW
  n_batch = total // args.batch_size
  y: int = Y[0].item()

  batch_AX, batch_idx = [], []
  def process_batch():
    nonlocal batch_AX, batch_idx
    assert len(batch_AX) == len(batch_idx) 

    AX = torch.concat(batch_AX, dim=0)
    logit = model(normalize(AX))
    prob  = F.softmax(logit, dim=-1)
    loss  = F.cross_entropy(logit, Y.expand([AX.shape[0]]), reduction='none')
    pred  = logit.argmax(dim=-1)

    for idx, (i, j) in enumerate(batch_idx):
      logits [i, j] = logit[idx][y].item()
      probs  [i, j] = prob [idx][y].item()
      losses [i, j] = loss [idx]   .item()
      preds  [i, j] = pred [idx]   .item()

    batch_AX .clear()
    batch_idx.clear()

  pbar = tqdm(total=n_batch)
  for i in range(qH):
    for j in range(qW):
      AX = X.clone()
      AX[:, :, i:i+pH, j:j+pW] = P
      batch_AX .append(AX)
      batch_idx.append((i, j))

      if len(batch_idx) >= args.batch_size:
        process_batch()
        pbar.update()

  if len(batch_idx):
    process_batch()
    pbar.update()
  pbar.close()

  expanme = f'grid_query-img={args.idx}_ptc={args.ip_idx}_r={args.ratio}'
  
  plt.subplot(221) ; plt.title('logits')  ; sns.heatmap(logits)
  plt.subplot(222) ; plt.title('probs')   ; sns.heatmap(probs, vmin=0.0, vmax=1.0)
  plt.subplot(223) ; plt.title('losses')  ; sns.heatmap(losses)
  plt.subplot(224) ; plt.title('correct') ; sns.heatmap(preds, vmin=0, vmax=999)
  plt.savefig(LOG_PATH / f'{expanme}.png', dpi=600)

  with lzma.open(LOG_PATH / f'{expanme}.xz.pkl', 'wb') as fh:
    pkl.dump({
      'losses': losses,
      'logits': logits,
      'probs':  probs,
      'preds':  preds,
    }, fh)


def go(args):
  model = get_model(args.model).to(device)
  model.eval()

  dataloader = get_dataloader(bs=1)
  for i, (X, Y) in enumerate(dataloader):
    if i >= args.idx: break
  X = X.to(device)
  Y = Y.to(device)

  with gzip.open(PACTCH_FILE, 'rb') as f:
    patches, targets, info = pkl.load(f)
  patch_size: int = info['patch_size']
  patch: Tensor = patches[args.ip_idx].unsqueeze(0).to(device)   # [B=1, C=3, H=50, W=50]
  B, C, H, W = patch.shape
  x, y = (H - patch_size) // 2, (W - patch_size) // 2
  patch = patch[:, :, x:x+patch_size, y:y+patch_size]
  scale: float = 224 * args.scale / patch_size
  P = F.interpolate(patch, scale_factor=scale, mode='nearest')

  run(args, model, X, Y, P)


if __name__ == '__main__':
  parser = ArgumentParser()
  # model & data
  parser.add_argument('-M', '--model', default='resnet50', choices=MODELS, help='model to attack')
  parser.add_argument('--batch_size', default=512, type=int)
  parser.add_argument('--idx', default=0, type=int, help='run image sample index')
  # ImageNet-patch (attack)
  parser.add_argument('--ratio', default=0.05, type=float, help='attack patch area ratio, typically 1%~5%')
  parser.add_argument('--ip_idx', default=0,   type=int,   help='which pre-gen patch to use (index range 0 ~ 9)')
  args = parser.parse_args()

  assert 0.0 < args.ratio < 1.0
  args.scale = np.sqrt(args.ratio)   # area to side

  if device == 'cuda':
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.ipc_collect()

  t = time()
  go(args)
  print(f'>> done in {time() - t:.3f}s')

  if device == 'cuda':
    alloc = torch.cuda.max_memory_allocated() // 2**20
    resrv = torch.cuda.max_memory_reserved()  // 2**20
    print(f'[vram] alloc: {alloc} MB, resrv: {resrv} MB')
