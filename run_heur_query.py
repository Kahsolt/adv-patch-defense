#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/24 

import random

from torch.nn import AvgPool2d
import matplotlib.pyplot as plt
import seaborn as sns
from moviepy.editor import ImageClip, concatenate_videoclips, ImageSequenceClip

from run_grid_query import *

npimg = np.ndarray

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

    pred = model(normalize(AX)).argmax(dim=-1)[0].item()
    imgs.append(tensor_to_plt(AX, title=f'query={cnt}; truth={truth}, pred={pred}'))
    if pred != truth: break
    cnt += 1

  expanme = f'rand_query-{args.model}_img={args.idx}_ptc={args.ip_idx}_r={args.ratio}'
  save_video(imgs, args.out_path / f'{expanme}.mp4')
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
    logits = model(normalize(AX))
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

  expanme = f'heur_query-{args.model}_img={args.idx}_ptc={args.ip_idx}_r={args.ratio}'
  save_video(imgs, args.out_path / f'{expanme}.mp4')
  return pred != truth, cnt


@torch.no_grad()
def run(args, model:Module, X:Tensor, Y:Tensor, P:Tensor):
  with torch.inference_mode():
    # raw pred
    y: int = Y[0].item()
    y_hat: int = model(normalize(X)).argmax(dim=-1)[0].item()
    print(f'>> truth: {y}, raw_pred: {y_hat}')

  with torch.no_grad():
    # heur query
    ok, cnt = heur_query(args, model, X, Y, P)
    print(f'>> [heur_query] success: {ok}, query times: {cnt}')
  
  with torch.inference_mode():
    # rand query
    ok, cnt = rand_query(args, model, X, Y, P)
    print(f'>> [rand_query] success: {ok}, query times: {cnt}')


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
  for i, (X, Y) in enumerate(dataloader):
    if i >= args.idx: break
  X = X.to(device)
  Y = Y.to(device)
  run(args, model, X, Y, P)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet50', choices=MODELS, help='model to attack')
  parser.add_argument('--batch_size',  default=512,  type=int,   help='patch attack query batch size')
  parser.add_argument('--idx',         default=0,    type=int,   help='run image sample index')
  parser.add_argument('--query',       default=500,  type=int,   help='max query count')
  parser.add_argument('--ratio',       default=0.05, type=float, help='attack patch area ratio, typically 1%~5%')
  parser.add_argument('--ip_idx',      default=0,    type=int,   help='which pre-gen patch to use (index range 0 ~ 9)')
  parser.add_argument('--hq_alpha',    default=2,    type=int,   help='heur_query shift step size')
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
