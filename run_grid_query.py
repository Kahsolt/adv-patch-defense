#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/24 

import gzip
import lzma
import pickle as pkl
from pathlib import Path
from time import time
from argparse import ArgumentParser
from typing import List

from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import grad
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

if 'repos':
  import sys
  sys.path.append('repo/ImageNet-Patch')
  from transforms.apply_patch import ApplyPatch
  PACTCH_FILE = "repo/ImageNet-Patch/assets/imagenet_patch.gz"

from utils import *

BASE_PATH = Path(__file__).parent
LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)


def saliency_map(model:Module, X:Tensor, Y:Tensor) -> np.ndarray:
  assert X.shape[0] == len(Y) == 1
  X.requires_grad = True
  logits = model(normalize(X))
  loss = F.cross_entropy(logits, Y, reduction='none')
  g = grad(loss, X, loss)[0]      # [B, H, W]
  g = g.abs().max(dim=1).values   # [B, C, H, W]
  return g[0].cpu().numpy().astype(np.float16)

def grid_query(model:Module, X:Tensor, Y:Tensor, P:Tensor) -> List[np.ndarray]:
  B, C, H, W = X.shape ; assert B == 1
  pB, pC, pH, pW = P.shape ; assert C == pC
  shape = qH, qW = [H - pH + 1, W - pW + 1]
  logits = np.zeros(shape, dtype=np.float16)
  probs  = np.zeros(shape, dtype=np.float16)
  losses = np.zeros(shape, dtype=np.float16)
  preds  = np.zeros(shape, dtype=np.uint16)

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
      logits[i, j] = logit[idx][y].item()
      probs [i, j] = prob [idx][y].item()
      losses[i, j] = loss [idx]   .item()
      preds [i, j] = pred [idx]   .item()

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

  return logits, probs , losses, preds


@torch.no_grad()
def run(args, model:Module, X:Tensor, Y:Tensor, P:Tensor):
  with torch.inference_mode():
    # raw pred
    y: int = Y[0].item()
    y_hat: int = model(normalize(X)).argmax(dim=-1)[0].item()
    print(f'>> truth: {y}, raw_pred: {y_hat}')

    # grid query
    logits, probs, losses, preds = grid_query(model, X, Y, P)
    correct: np.ndarray = preds == y
    asr: float = 1.0 - correct.sum() / correct.size
    print(f'>> random asr: {asr:.3%}')

  with torch.enable_grad():
    # saliency map
    smap = saliency_map(model, X, Y)

  expanme = f'grid_query-{args.model}_img={args.idx}_ptc={args.ip_idx}_r={args.ratio}'
  suptitle = f'{args.model}_img={args.idx}_ptc={args.ip_idx}_r={args.ratio} (asr: {asr:.3%})'

  plt.figure(figsize=(16, 10))
  plt.subplot(231) ; plt.title('logits')   ; sns.heatmap(logits)
  plt.subplot(234) ; plt.title('probs')    ; sns.heatmap(probs, vmin=0.0, vmax=1.0)
  plt.subplot(232) ; plt.title('losses')   ; sns.heatmap(losses)
  plt.subplot(235) ; plt.title('saliency') ; sns.heatmap(smap)
  plt.subplot(233) ; plt.title('preds')    ; sns.heatmap(preds, vmin=0, vmax=999)
  plt.subplot(236) ; plt.title('correct')  ; sns.heatmap(correct, vmin=0, vmax=1)
  plt.suptitle(suptitle)
  plt.tight_layout()
  fp = args.out_path / f'{expanme}.png'
  plt.savefig(fp, dpi=600)
  print(f'>> save figure to {fp}')

  fp = args.out_path / f'{expanme}.xz.pkl'
  print(f'>> save results to {fp}')
  with lzma.open(fp, 'wb') as fh:
    pkl.dump({
      'truth':  y,
      'pred':   y_hat,
      'logits': logits,
      'probs':  probs,
      'losses': losses,
      'preds':  preds,
      'smap':   smap,
      'asr':    asr,
    }, fh)


def go(args):
  model = get_model(args.model).to(device)
  model.eval()

  with gzip.open(PACTCH_FILE, 'rb') as f:
    patches, targets, info = pkl.load(f)
  patch_size: int = info['patch_size']
  patch: Tensor = patches[args.ip_idx].unsqueeze(0).to(device)   # [B=1, C=3, H=50, W=50]
  B, C, H, W = patch.shape
  x, y = (H - patch_size) // 2, (W - patch_size) // 2
  patch = patch[:, :, x:x+patch_size, y:y+patch_size]
  scale: float = 224 * args.scale / patch_size
  P = F.interpolate(patch, scale_factor=scale, mode='nearest')

  dataloader = get_dataloader(bs=1)
  if args.idx_all:
    for i, (X, Y) in enumerate(dataloader):
      print(f'[{i} / {len(dataloader)}]')
      X = X.to(device)
      Y = Y.to(device)
      run(args, model, X, Y, P)
  else:
    for i, (X, Y) in enumerate(dataloader):
      if i >= args.idx: break
    X = X.to(device)
    Y = Y.to(device)
    run(args, model, X, Y, P)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet50', choices=MODELS, help='model to attack')
  parser.add_argument('--batch_size', default=512, type=int, help='patch attack query batch size')
  parser.add_argument('--idx', default=0, type=int, help='run image sample index')
  parser.add_argument('--idx_all', action='store_true', help='run all test images')
  parser.add_argument('--ratio', default=0.05, type=float, help='attack patch area ratio, typically 1%~5%')
  parser.add_argument('--ip_idx', default=0,   type=int,   help='which pre-gen patch to use (index range 0 ~ 9)')
  parser.add_argument('--ip_idx_all', action='store_true', help='run all pre-generated patches')
  parser.add_argument('--out_path', default=LOG_PATH, type=Path, help='out data path')
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
