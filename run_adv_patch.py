#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/24 

from argparse import ArgumentParser
from functools import partial
from types import MethodType

from tqdm import tqdm
from torch.nn import Module, CrossEntropyLoss, AvgPool2d
import torch.nn.functional as F
from torch.autograd import Function as TorchFunction
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AdversarialPatchPyTorch
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch

from data import *
from model import *
from utils import *

if 'repos':
  sys.path.append(str(IP_PATH))
  from transforms.apply_patch import ApplyPatch
  sys.path.append(str(SAC_PATH))
  from patch_detector import PatchDetector
  sys.path.append(str(MAE_PATH))
  from models_mae import MaskedAutoencoderViT, mae_vit_large_patch16


class ThresholdSTEFunction(TorchFunction):
  @staticmethod
  def forward(ctx, input):
    global threshold
    return (input > threshold).float()

  @staticmethod
  def backward(ctx, grad):
    return grad


''' concrete DFN & ATK '''

def get_sac(args) -> PatchDetector:
  ps = args.patch_size
  sac = PatchDetector(3, 1, base_filter=16, square_sizes=[int(ps*1.5), ps, ps//2], n_patch=1)
  sac.unet.load_state_dict(torch.load(SAC_CKPT, map_location=device))
  return sac.eval().to(device)

def get_mae(args) -> MaskedAutoencoderViT:
  mae = mae_vit_large_patch16()
  mae.load_state_dict(torch.load(MAE_CKPT, map_location=device)['model'])
  mae = mae.eval().to(device)

  def recover_masked_one(self:MaskedAutoencoderViT, x:Tensor, x_mask:Tensor) -> Tensor:
    # NOTE: we cannot handle batch, due to possible length mismatch
    assert x.shape[0] == 1
    x_orig = x

    ''' .forward_encoder() '''
    # embed patches
    x = self.patch_embed(x)
    # add pos embed w/o cls token
    x = x + self.pos_embed[:, 1:, :]

    # masking: length -> length * mask_ratio
    def known_masking(x:Tensor, x_mask:Tensor):
      B, L, D = x.shape
      B, C, H, W = x_mask.shape
      assert C == 1 and L == H * W

      mask = x_mask.reshape((B, C*H*W))             # [B=1, L]
      ids_shuffle = torch.argsort(mask, dim=1)      # ascend: 0 is keep, 1 is remove
      ids_restore = torch.argsort(ids_shuffle, dim=1)
      len_keep = (mask[0] == 0).sum()
      ids_keep = ids_shuffle[:, :len_keep]
      # [B=1, L'<=196, D=1024]
      x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
      return x_masked, mask, ids_restore

    x, mask, ids_restore = known_masking(x, x_mask)

    # append cls token
    cls_token = self.cls_token + self.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # apply Transformer blocks
    for blk in self.blocks:
      x = blk(x)
    x = self.norm(x)

    z = self.forward_decoder(x, ids_restore)  # [N, L, p*p*3]
    y_hat = self.unpatchify(z)                # [N, C, H, W]

    # paste predicted area to known
    mask = mask.unsqueeze(-1).repeat(1, 1, self.patch_embed.patch_size[0]**2*3)  # (N, H*W, p*p*3)
    mask = self.unpatchify(mask)              # 1 is removing, 0 is keeping
    return x_orig * (1 - mask) + y_hat * mask

  def recover_masked(self:MaskedAutoencoderViT, x:Tensor, x_mask:Tensor) -> Tensor:
    res = []
    for i in range(x.shape[0]):
      xi = recover_masked_one(self, x[i:i+1, ...], x_mask[i:i+1, ...])
      res.append(xi.cpu())
    return torch.cat(res, dim=0).to(x.device)

  mae.recover_masked = MethodType(recover_masked, mae)

  def ci_random_masking(self:MaskedAutoencoderViT, x, n_splits=4):
    """ x: [N, L, D], the patch sequence """
    assert isinstance(n_splits, int) and n_splits >= 2

    # [B=1, L=196=14*14, D=1024]
    N, L, D = x.shape
    import math
    k = math.ceil(L / n_splits)         # masked patch count

    # [B=1, L=196]
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    # [B=1, L=196], sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    # [B=1, L=196]
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # gather all non-overlapping subsets
    subsets = []
    for i in range(n_splits):
      # pick out one subset
      slicer_L = slice(None), slice(None, k*i)            # leave (k-1)/k untouched
      slicer_M = slice(None), slice(k*i, k*(i+1))         # mask 1/k patches
      slicer_R = slice(None), slice(k*(i+1), None)
      # [B=1, (k-1)/k*L]
      ids_keep = torch.cat([ids_shuffle[slicer_L], ids_shuffle[slicer_R]], axis=-1)
      #ids_keep = ids_shuffle[slicer_M]
      # [B=1, (k-1)/k*L, D=1024]
      x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
      # [B, L], generate the binary mask: 0 is keep, 1 is remove
      mask = torch.zeros([N, L], device=x.device)
      mask[slicer_M] = 1
      # unshuffle to get the binary mask
      mask = torch.gather(mask, dim=1, index=ids_restore)
      # one split
      subsets.append((x_masked, mask, k*i))

    # subsets[0][0]: [1, 171/175, 1024], use major to predict minor
    # subsets[0][1]: [1, 196], non-overlap, sum(subsets[i][1]) == 196
    # ids_restore:   [1, 196], permutation of range [0, 195]
    return subsets, ids_restore

  def ci_forward_splitter(self:MaskedAutoencoderViT, x, n_splits):
    # embed patches
    x = self.patch_embed(x)
    # add pos embed w/o cls token
    x = x + self.pos_embed[:, 1:, :]
    # masking split n overlapping subsets
    return self.ci_random_masking(x, n_splits)

  def ci_forward_encoder(self:MaskedAutoencoderViT, x):
    # append cls token
    cls_token = self.cls_token + self.pos_embed[:, :1, :]
    cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    # apply Transformer blocks
    for blk in self.blocks: x = blk(x)
    return self.norm(x)

  def ci_forward_decoder(self:MaskedAutoencoderViT, x, ids_restore, k):
    # embed tokens
    x = self.decoder_embed(x)
    # append mask tokens to sequence
    mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    x_ = x[:, 1:, :]      # no cls token
    x_ = torch.cat([x_[:, :k, :], mask_tokens, x_[:, k:, :]], dim=1)
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
    # add pos embed
    x = x + self.decoder_pos_embed
    # apply Transformer blocks
    for blk in self.decoder_blocks: x = blk(x)
    x = self.decoder_norm(x)
    # predictor projection
    x = self.decoder_pred(x)
    # remove cls token
    return x[:, 1:, :]

  def ci_forward(self:MaskedAutoencoderViT, x, n_splits=4):
    subsets, ids_restore = self.ci_forward_splitter(x, n_splits)

    preds, masks = [], []
    for z, mask, k in subsets:
      latent = self.ci_forward_encoder(z)
      pred = self.ci_forward_decoder(latent, ids_restore, k)  # [N, L, p*p*3]
      preds.append(pred)
      masks.append(mask)
    return preds, masks

  def cross_infer(self:MaskedAutoencoderViT, x:Tensor, n_split:int=8) -> Tensor:
    preds, masks = self.ci_forward(x, n_split)

    y = torch.zeros_like(preds[0])
    for p, m in zip(preds, masks):
      y = y + p * m.unsqueeze(-1)     # 1 is masked areas for prediction to fill
    return self.unpatchify(y)

  mae.ci_random_masking   = MethodType(ci_random_masking,   mae)
  mae.ci_forward_splitter = MethodType(ci_forward_splitter, mae)
  mae.ci_forward_encoder  = MethodType(ci_forward_encoder,  mae)
  mae.ci_forward_decoder  = MethodType(ci_forward_decoder,  mae)
  mae.ci_forward          = MethodType(ci_forward,          mae)
  mae.cross_infer         = MethodType(cross_infer,         mae)

  return mae

def get_ap(args, clf:Module) -> AdversarialPatchPyTorch:
  if args.ap_pgd:
    optimizer     = "pgd"
    learning_rate = 1/255
  else:
    optimizer     = "Adam"
    learning_rate = 5.0

  return AdversarialPatchPyTorch(
    estimator=clf,
    rotation_max=args.ap_rot,
    scale_min=args.scale,
    scale_max=args.scale,
    distortion_scale_max=0.0,
    optimizer=optimizer,
    learning_rate=learning_rate,
    max_iter=args.ap_iter,
    batch_size=1,
    patch_shape=(3, args.patch_size, args.patch_size),
    patch_location=None,
    patch_type=args.ap_shape,
    targeted=False,
    summary_writer=False,
    verbose=True,
  )

def get_ip(args) -> AdversarialPatchPyTorch:
  import gzip, pickle
  with gzip.open(IP_FILE, 'rb') as f:
    patches, targets, info = pickle.load(f)
  patch_size: int = info['patch_size']
  patch: Tensor = patches[args.ip_idx]    # 224x224 的黑色画布中心有个 50x50 的 patch
  scale: float = args.patch_size / patch_size

  apply_patch = ApplyPatch(
    patch,
    translation_range=(args.ip_tx, args.ip_tx),   # translation fraction wrt image dimensions
    rotation_range=args.ip_rot,     # maximum absolute value of the rotation in degree
    scale_range=(scale, scale),     # scale range wrt image dimensions
  )

  class ImageNetPatchPyTorch(Module):
    def generate(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
      patch_np = patch.unsqueeze(dim=0).cpu().numpy()
      return patch_np, np.zeros_like(patch_np)
    def apply_patch(self, x:np.ndarray, ignored:float) -> np.ndarray:
      return apply_patch(torch.from_numpy(x)).cpu().numpy()

  return ImageNetPatchPyTorch()

''' abstract DFN & ATK '''

def get_dfn(args) -> PreprocessorPyTorch:
  sac = get_sac(args) if args.sac else None
  mae = get_mae(args) if args.mae else None
  if not any([sac, mae]): return None

  class Defenser(PreprocessorPyTorch):
    
    AVG_POOL = AvgPool2d(MAE_PATCH_SIZE, MAE_PATCH_SIZE)

    def forward(self, AX:Tensor, Y:Tensor) -> Tuple[Tensor, Tensor]:
      return self.forward_show(AX, Y, show=False)
    
    def forward_show(self, AX:Tensor, Y:Tensor, show:bool=True) -> Tuple[Tensor, Tensor]:
      if 'resize':
        B, C, H, W = AX.shape
        if not (H == W == 224):
          AX = F.interpolate(AX, size=(224, 224), mode='nearest')
          resized = True
        else:
          resized = False

      if sac:
        # `mask` is a round-corner rectangle when `shape_completion=True`; values are binarized to {0, 1}
        # `bpda` enables gradient bypassing
        RX, masks, _ = sac(AX, bpda=True, shape_completion=args.sac_complete)
        RX = torch.stack(RX, dim=0)
        if show: imshow_torch(AX, RX, title='sac')
        AX = RX

      if mae:
        if sac:
          masks = torch.cat(masks, dim=0)
          pmasks = self.AVG_POOL(masks)
          pmasks = ThresholdSTEFunction.apply(pmasks)
          if show: imshow_torch(masks, pmasks, title='mask')
          RX = denormalizer(mae.recover_masked(normalizer(AX), pmasks))
        else:
          RX = denormalizer(mae.cross_infer(normalizer(AX), args.mae_split))
        if show: imshow_torch(AX, RX, title='mae')
        AX = RX

      if 'unresize':
        if resized:
          RX = F.interpolate(RX, size=(H, W), mode='nearest')

      return RX, Y

  return Defenser()

def get_atk(args, model:Module, dfn:PreprocessorPyTorch=None) -> AdversarialPatchPyTorch:
  if not any([args.ap, args.ip]): return
  assert not all([args.ap, args.ip]), 'should specify at most one attack method'

  clf = PyTorchClassifier(
    model=model,
    loss=CrossEntropyLoss(),
    input_shape=(args.img_size, args.img_size, 3),
    nb_classes=args.n_class,
    optimizer=None,
    clip_values=(0.0, 1.0),
    preprocessing_defences=dfn,
    postprocessing_defences=None,
    preprocessing=DATASET_STATS[args.dataset],   # normalize
    device_type="gpu",
  )
  
  if args.ap: return get_ap(args, clf)
  if args.ip: return get_ip(args)


@torch.no_grad()
def run(args, model:Module, dataloader:DataLoader, atk:AdversarialPatchPyTorch=None, dfn:PreprocessorPyTorch=None) -> float:
  def run_one_batch(X:Tensor, Y:Tensor):
    nonlocal total, correct

    if not atk:
      pred = model(normalizer(X)).argmax(dim=-1)
      correct += (pred == Y).sum()
      total += len(pred)
      return

    # adv generation
    with torch.enable_grad():
      X_np = X.cpu().numpy()
      Y_np = Y.cpu().numpy()
      P_np, M_np = atk.generate(X_np, Y_np)
      if args.show: imshow_torch(torch.from_numpy(P_np), torch.from_numpy(M_np), title='adv-patch')

    # random query
    succ = torch.zeros_like(Y).bool()   # 允许查询次数内有一次攻击成功就算成功
    for i in range(args.query):
      AX = torch.from_numpy(atk.apply_patch(X_np, args.scale)).to(device)
      if args.show: imshow_torch(X, AX, title='atk')
      if dfn: AX, Y = dfn.forward_show(AX, Y, args.show)

      pred = model(normalizer(AX)).argmax(dim=-1)
      succ |= (pred != Y)

      if i % args.log_interval == 0: print(f'[query {i} / {args.query}] asr: {succ.sum()/ len(succ):.3%}')
      if succ.all(): break    # stop early

    print(f'[final {i}] asr: {succ.sum()/ len(succ):.3%}')
    
    if args.show_adv: imshow_torch(X, AX, title='atk')
    correct += (~succ).sum().item()
    total += len(pred)

    gc_all()    # do not know why :(

  model.eval()
  total, correct = 0, 0

  if args.idx:      # run single image
    X, Y = dataloader.dataset[args.idx]
    X = X.unsqueeze(dim=0).to(device)
    Y = Tensor([Y]).long().unsqueeze(dim=0).to(device)
    run_one_batch(X, Y)
  else:             # run total dataset
    for X, Y in tqdm(dataloader):
      if args.limit and total >= args.limit: break
      X = X.to(device)
      Y = Y.to(device)
      run_one_batch(X, Y)

  return correct / total


@perf_count
def go(args):
  if args.dataset == 'cifar10':
    model = get_model_pytorch_cifar10(args.model).to(device)
  else:
    model = get_model(args.model).to(device)

  dfn = get_dfn(args)
  atk = get_atk(args, model, dfn)

  dataloader = get_dataloader(args.dataset, split='test', batch_size=(1 if atk else 32))
  acc = run(args, model, dataloader, atk, dfn)
  print(f'Accuracy: {acc:.3%}')


if __name__ == '__main__':
  parser = ArgumentParser()
  # model & data
  parser.add_argument('-M', '--model',      default='resnet50', help='model to attack')
  parser.add_argument('-D', '--dataset',    default='imagenet', choices=DATASETS)
  parser.add_argument('-L', '--limit',      default=0,  type=int, help='limit run sample count')
  parser.add_argument('-I', '--idx',                    type=int, help='run sample index')
  # patch-like attacks common
  parser.add_argument('-S', '--patch_size', default=0,   type=int, help='attack patch size, default 7 for 32x32 and 50 for 224x224 (typical area ratio 5%)')
  parser.add_argument('-Q', '--query',      default=500, type=int, help='attack query time limit')
  # adv-patch (attack)
  parser.add_argument('--ap',       action='store_true', help='enable adv-patch attack')
  parser.add_argument('--ap_shape', default='square', choices=['circle', 'square'], help='patch shape')
  parser.add_argument('--ap_rot',   default=22.5, type=float, help='max patch rotation angle')
  parser.add_argument('--ap_pgd',   action='store_true',      help='optim method, pgd or simple Adam')
  parser.add_argument('--ap_iter',  default=100,  type=int,   help='optim iter')
  # ImageNet-patch (attack)
  parser.add_argument('--ip',     action='store_true',     help='enable ImageNet-patch attack')
  parser.add_argument('--ip_idx', default=0,   type=int,   help='which pre-gen patch to use (index range 0 ~ 9)')
  parser.add_argument('--ip_rot', default=45,  type=float, help='max patch rotation angle')
  parser.add_argument('--ip_tx',  default=0.2, type=float, help='max patch translation fraction')
  # sac (defense)
  parser.add_argument('--sac',          action='store_true', help='enable SAC defense')
  parser.add_argument('--sac_complete', action='store_true', help='enable shape completion')
  # mae (defense)
  parser.add_argument('--mae',        action='store_true', help='enable MAE defense')
  parser.add_argument('--mae_split',  default=4, type=int, help='if w/o --sac, cross infer n_split')
  parser.add_argument('--mae_thresh', default=4, type=int, help='if w/ --sac, only mask those pathes with bad pixels count exceeding threshold')
  # debug
  parser.add_argument('--show',     action='store_true', help='show debug images')
  parser.add_argument('--show_adv', action='store_true', help='show final adv images')
  parser.add_argument('--log_interval', default=100, type=int, help='show ASR log during attack querying')
  args = parser.parse_args()

  if args.dataset == 'cifar10':
    assert not args.ip, 'should not put --ip over cifar10 :('
    assert args.model in PYTORCH_CIFAR10_MODELS, f'model must choose from {PYTORCH_CIFAR10_MODELS}'
    args.n_class = 10
    args.img_size = 32
    args.patch_size = args.patch_size or 7
  else:
    assert args.model in TORCHVISION_MODELS, f'model must choose from {TORCHVISION_MODELS}'
    args.n_class = 1000
    args.img_size = 224
    args.patch_size = args.patch_size or 50
  args.scale = args.patch_size / args.img_size   # side

  assert 0 < args.mae_split <= 36
  assert 0 <= args.ip_idx <= 9

  threshold = args.mae_thresh / MAE_PATCH_SIZE ** 2

  normalizer   = partial(normalize,   args.dataset)
  denormalizer = partial(denormalize, args.dataset)

  go(args)
