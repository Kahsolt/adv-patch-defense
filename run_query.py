#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/24 

import gzip
import lzma
import random
import pickle as pkl
from pathlib import Path
from functools import partial
from argparse import ArgumentParser

import numpy as np
from torch.nn import AvgPool2d
import torch.nn.functional as F
from torch.autograd import grad
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from moviepy.editor import ImageClip, concatenate_videoclips, ImageSequenceClip

from data import *
from model import *
from utils import *

LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)


def tensor_to_plt(X:Tensor, title:str='') -> npimg:
  im = (X[0].permute([1, 2, 0]).clip(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)

  plt.clf()
  fig, ax = plt.subplots()
  ax.imshow(im)
  ax.axis('off')
  fig.suptitle(title)
  cvs = fig.canvas
  cvs.draw()
  im = np.frombuffer(cvs.tostring_rgb(), dtype=np.uint8).reshape(*reversed(cvs.get_width_height()), 3)
  plt.close('all')
  return im

def save_video(imgs:List[npimg], fp:str, fps:int=8):
  if not len(imgs): return
  imgs_ext = imgs + [imgs[-1]]*fps*2
  try:
    clip = ImageSequenceClip(imgs_ext, fps=fps)
  except:
    frames = [ImageClip(img).set_duration(1/fps) for img in imgs_ext]
    clip = concatenate_videoclips(frames, method='compose')
  finally:
    clip.write_videofile(str(fp), fps=fps)

def get_expname(args) -> str:
  return f'{args.model}_img={args.idx}_ptc={args.ip_idx}_ps={args.patch_size}'


def saliency_map(model:Module, X:Tensor, Y:Tensor) -> np.ndarray:
  assert X.shape[0] == len(Y) == 1
  X.requires_grad = True
  logits = model(normalizer(X))
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
    logit = model(normalizer(AX))
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

  return logits, probs, losses, preds

def rand_query(args, model:Module, X:Tensor, Y:Tensor, P:Tensor) -> Tuple[bool, int]:
  B, C, H, W = X.shape ; assert B == 1
  pB, pC, pH, pW = P.shape ; assert C == pC
  qH, qW = H - pH + 1, W - pW + 1
  truth: int = Y[0].item()
  
  imgs = []
  cnt = 1
  while cnt < args.query:
    x = random.randrange(0, qH)
    y = random.randrange(0, qW)
    AX = X.clone()
    AX[:, :, x:x+pH, y:y+pW] = P

    pred = model(normalizer(AX)).argmax(dim=-1)[0].item()
    imgs.append(tensor_to_plt(AX, title=f'query={cnt}; truth={truth}, pred={pred}'))
    if pred != truth: break
    cnt += 1

  save_video(imgs, args.out_path / f'rand_query-{get_expname(args)}.mp4')
  return pred != truth, cnt

def heur_query(args, model:Module, X:Tensor, Y:Tensor, P:Tensor) -> Tuple[bool, int]:
  B, C, H, W = X.shape ; assert B == 1
  pB, pC, pH, pW = P.shape ; assert C == pC
  qH, qW = H - pH + 1, W - pW + 1
  truth: int = Y[0].item()

  visited = set()
  imgs = []
  cnt = 1
  def paste_and_query(x:int, y:int) -> Tuple[int, float]:
    #if (x, y) in visited: raise ValueError
    nonlocal cnt ; cnt += 1
    visited.add((x, y))
    AX = X.clone()
    AX[:, :, x:x+pH, y:y+pW] = P
    logits = model(normalizer(AX))
    pred = logits.argmax(dim=-1)[0].item()
    imgs.append(tensor_to_plt(AX, title=f'query={cnt}; truth={truth}, pred={pred}'))
    loss = F.cross_entropy(logits, Y).item()
    return pred, loss
  
  # oneshot attack (?)
  with torch.enable_grad(): smap = saliency_map(model, X, Y)
  smap = torch.from_numpy(smap).float().unsqueeze(0).unsqueeze(0)
  scores: Tensor = AvgPool2d(kernel_size=(pH, pW), stride=1)(smap).squeeze()  # HW = 224 - 50 + 1
  if not 'debug smap':
    plt.clf()
    plt.subplot(121) ; sns.heatmap(smap[0][0].cpu().numpy())
    plt.subplot(122) ; sns.heatmap(scores    .cpu().numpy())
    plt.show()

  sc_pos = []
  for i in range(scores.shape[0]):
    for j in range(scores.shape[1]):
      sc = scores[i, j].item()
      pos = i, j
      sc_pos.append((sc, pos))
  sc_pos = sorted(sc_pos, reverse=True)
  sc, (x, y) = sc_pos[0] ; assert sc == scores.max().item()
  pred, loss = paste_and_query(x, y)
  momentum = np.asarray([0.0, 0.0])

  def _spos(x, y) -> Tuple[int, int]:
    x = max(0, min(x, qH-1))
    y = max(0, min(y, qW-1))
    return x, y

  def rand_pos() -> Tuple[int, int]:
    nonlocal visited
    while True:
      x = random.randrange(0, qH)
      y = random.randrange(0, qW)
      pos = (x, y)
      if pos not in visited: return pos

  while True:
    if pred != truth: break
    if cnt >= args.query: break

    print('momentum:', momentum)
    
    if np.allclose(momentum, np.asarray([0.0, 0.0]), atol=1e-5):    # random move on grad vanish
      pos_neighbor = [
        #_spos(x-args.hq_alpha, y              ),
        #_spos(x+args.hq_alpha, y              ),
        #_spos(x,               y-args.hq_alpha),
        #_spos(x,               y+args.hq_alpha),
        _spos(x-args.hq_alpha, y-args.hq_alpha),
        _spos(x-args.hq_alpha, y+args.hq_alpha),
        _spos(x+args.hq_alpha, y-args.hq_alpha),
        _spos(x+args.hq_alpha, y+args.hq_alpha),
      ]
      pos_cand = list({pos for pos in pos_neighbor if pos not in visited})

      if len(pos_cand) == 0:    # random restart on dead end
        x, y = rand_pos()
        pred, loss = paste_and_query(x, y)
        momentum = np.asarray([0.0, 0.0])
      else:
        nx, ny = random.choice(pos_cand)
        pred, nloss = paste_and_query(nx, ny)
        if 'update momentum':
          dx, dy = nx - x, ny - y
          dloss = nloss - loss
          momentum[0] = 0.1 * momentum[0] + 0.9 * ((dloss / 2 / dx) if dx else 0.0)
          momentum[1] = 0.1 * momentum[1] + 0.9 * ((dloss / 2 / dy) if dy else 0.0)
        loss = nloss
        x, y = nx, ny
    else:
      nx, ny = _spos(x + int(np.sign(momentum[0])) * args.hq_alpha, y + int(np.sign(momentum[1])) * args.hq_alpha)
      pred, nloss = paste_and_query(nx, ny)
      if 'update momentum':
        dx, dy = nx - x, ny - y
        dloss = nloss - loss
        momentum[0] = 0.1 * momentum[0] + 0.9 * ((dloss / 2 / dx) if dx else 0.0)
        momentum[1] = 0.1 * momentum[1] + 0.9 * ((dloss / 2 / dy) if dy else 0.0)
      loss = nloss
      x, y = nx, ny

  save_video(imgs, args.out_path / f'heur_query-{get_expname(args)}.mp4')
  return pred != truth, cnt


@torch.no_grad()
def run(args, model:Module, X:Tensor, Y:Tensor, P:Tensor):
  with torch.inference_mode():
    y: int = Y[0].item()
    y_hat: int = model(normalizer(X)).argmax(dim=-1)[0].item()
    print(f'>> truth: {y}, raw_pred: {y_hat}')

  if args.mode == 'attack':
    with torch.no_grad():
      ok, cnt = heur_query(args, model, X, Y, P)
      print(f'>> [heur_query] success: {ok}, query times: {cnt}')
    
    with torch.inference_mode():
      ok, cnt = rand_query(args, model, X, Y, P)
      print(f'>> [rand_query] success: {ok}, query times: {cnt}')

  if args.mode == 'grid':
    with torch.enable_grad():
      smap = saliency_map(model, X, Y)

    with torch.inference_mode():
      logits, probs, losses, preds = grid_query(model, X, Y, P)
      correct: np.ndarray = preds == y
      asr: float = 1.0 - correct.sum() / correct.size
      print(f'>> [grid_query] random asr: {asr:.3%}')

      expname = get_expname(args)
      savename = f'grid_query-{expname}'
      suptitle = f'{expname} (asr: {asr:.3%})'

      plt.figure(figsize=(16, 10))
      plt.subplot(231) ; plt.title('logits')   ; sns.heatmap(logits)
      plt.subplot(234) ; plt.title('probs')    ; sns.heatmap(probs, vmin=0.0, vmax=1.0)
      plt.subplot(232) ; plt.title('losses')   ; sns.heatmap(losses)
      plt.subplot(235) ; plt.title('saliency') ; sns.heatmap(smap)
      plt.subplot(233) ; plt.title('preds')    ; sns.heatmap(preds, vmin=0, vmax=999)
      plt.subplot(236) ; plt.title('correct')  ; sns.heatmap(correct, vmin=0, vmax=1)
      plt.suptitle(suptitle)
      plt.tight_layout()
      fp = args.out_path / f'{savename}.png'
      plt.savefig(fp, dpi=600)
      print(f'>> save figure to {fp}')

      fp = args.out_path / f'{savename}.xz.pkl'
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


@perf_count
def go(args):
  model = get_model(args.model).to(device)
  model.eval()

  with gzip.open(IP_FILE, 'rb') as f:
    patches, targets, info = pkl.load(f)
  patch_size: int = info['patch_size']

  ip_idx_ls = range(10) if args.ip_idx_all else [args.ip_idx]
  for ip_idx in ip_idx_ls:
    print(f'>> apply patch {ip_idx} ...')
    
    patch: Tensor = patches[ip_idx].unsqueeze(0).to(device)   # [B=1, C=3, H=224, W=224]
    B, C, H, W = patch.shape
    x, y = (H - patch_size) // 2, (W - patch_size) // 2
    patch = patch[:, :, x:x+patch_size, y:y+patch_size]       # [B=1, C=3, H=50, W=50], center crop out the real patch
    P = F.interpolate(patch, size=(args.patch_size, args.patch_size), mode='nearest')

    dataloader = get_dataloader(args.dataset, batch_size=1)
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
  parser.add_argument('-F', '--mode',    default='attack',   choices=['attack', 'grid'], help='simulate attacks or make grid query data')
  parser.add_argument('-M', '--model',   default='resnet50', help='model to attack')
  parser.add_argument('-D', '--dataset', default='imagenet', choices=DATASETS)
  parser.add_argument('--query',         default=500,      type=int,  help='attack query count limit')
  parser.add_argument('--hq_alpha',      default=2,        type=int,  help='heur_query shift step size')
  parser.add_argument('--batch_size',    default=512,      type=int,  help='grid query batch size')
  parser.add_argument('--patch_size',    default=50,       type=int,  help='attack patch size, set 7 for 32x32 and 50 for 224x224 (typical area ratio 5%)')
  parser.add_argument('--idx',           default=0,        type=int,  help='run test image sample index')
  parser.add_argument('--idx_all',       action='store_true',         help='run all test images')
  parser.add_argument('--ip_idx',        default=0,        type=int,  help='run ImageNet-patch index (range 0 ~ 9)')
  parser.add_argument('--ip_idx_all',    action='store_true',         help='run all ImageNet-patches')
  parser.add_argument('--out_path',      default=LOG_PATH, type=Path, help='out data path')
  args = parser.parse_args()

  if args.dataset == 'cifar10':
    assert args.model in PYTORCH_CIFAR10_MODELS, f'model must choose from {PYTORCH_CIFAR10_MODELS}'
  else:
    assert args.model in TORCHVISION_MODELS, f'model must choose from {TORCHVISION_MODELS}'

  normalizer   = partial(normalize,   args.dataset)
  denormalizer = partial(denormalize, args.dataset)

  go(args)
