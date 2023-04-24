#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/04/24 

from types import MethodType
from argparse import ArgumentParser
import torch.nn as nn

import sys
sys.path.append('repo/adversarial-robustness-toolbox')
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AdversarialPatchPyTorch
from art.defences.preprocessor.preprocessor import PreprocessorPyTorch
sys.path.append('repo/SegmentAndComplete')
from patch_detector import PatchDetector, ThresholdSTEFunction
SAC_CKPT = "repo/SegmentAndComplete/ckpts/coco_at.pth"
sys.path.append('repo/mae')
from models_mae import MaskedAutoencoderViT, mae_vit_large_patch16
PATCH_SIZE = 16
MAE_CKPT = "repo/mae/models/mae_visualize_vit_large_ganloss.pth"

from utils import *

# AdvPatch 攻击分类/检测模型： SAC + MAE 防御


class SAC_Proprocessor(PreprocessorPyTorch):

  def forward(self, AX: Tensor, Y) -> tuple:
    RX, mask_list, raw_mask_list = sac(AX, bpda=True, shape_completion=False)
    RX = torch.stack(RX, dim=0)
    return RX, Y

class MAE_Proprocessor(PreprocessorPyTorch):

  def forward(self, AX: Tensor, Y) -> tuple:
    _, y_hat, _ = mae(normalize(AX), mask_ratio)
    RX = denormalize(mae.unpatchify(y_hat))
    return RX, Y

class SAC_MAE_Proprocessor(PreprocessorPyTorch):

  def forward(self, AX: Tensor, Y) -> tuple:
    RX, mask_list, raw_mask_list = sac(AX, bpda=True, shape_completion=False)
    RX = torch.stack(RX, dim=0)
    masks = torch.cat(raw_mask_list, dim=0)
    pmasks = nn.AvgPool2d(PATCH_SIZE, PATCH_SIZE)(masks)
    pmasks = ThresholdSTEFunction.apply(pmasks, args.sac_thresh)
    pmasks = pmasks.long()
    RX = denormalize(mae.recover_masked(normalize(AX), pmasks))
    return RX, Y


def get_sac() -> PatchDetector:
  sac = PatchDetector(3, 1, base_filter=16, square_sizes=[100, 75, 50, 25], n_patch=1)
  sac.unet.load_state_dict(torch.load(SAC_CKPT, map_location=device))
  return sac.eval().to(device)

def get_mae() -> MaskedAutoencoderViT:
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

  return mae

def get_atk(model:nn.Module, sac:bool=False, mae:bool=False) -> AdversarialPatchPyTorch:
  global args

  preprocessing_defences = None
  if sac and mae: preprocessing_defences = SAC_MAE_Proprocessor()
  elif       sac: preprocessing_defences = SAC_Proprocessor()
  elif       mae: preprocessing_defences = MAE_Proprocessor()
  print('preprocessing_defences:', preprocessing_defences)

  clf = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    input_shape=(224, 224, 3),
    nb_classes=1000,
    optimizer=None,
    clip_values=(0.0, 1.0),
    preprocessing_defences=preprocessing_defences,
    postprocessing_defences=None,
    preprocessing=imagenet_stats(),
    device_type="gpu",
  )

  if args.pgd:
    optimizer     = "pgd"
    learning_rate = 1/255
    max_iter      = 100
  else:
    optimizer     = "Adam"
    learning_rate = 5.0
    max_iter      = 500
  
  atk = AdversarialPatchPyTorch(
    estimator=clf,
    rotation_max=22.5,
    scale_min=0.1,
    scale_max=1.0,
    distortion_scale_max=0.0,
    optimizer=optimizer,
    learning_rate=learning_rate,
    max_iter=max_iter,
    batch_size=args.batch_size,
    patch_shape=(3, 224, 224),
    patch_location=None,
    patch_type=args.shape,
    targeted=False,
    summary_writer=False,
    verbose=True,
  )
  return atk


@torch.no_grad()
def test(model, dataloader, atk:AdversarialPatchPyTorch=None, sac:PatchDetector=None, mae:MaskedAutoencoderViT=None) -> float:
  global args
  total, correct = 0, 0

  model.eval()
  for X, Y in dataloader:
    X = X.to(device)
    Y = Y.to(device)

    if atk:
      with torch.enable_grad():
        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()
        P, M = atk.generate(X_np, Y_np)
        if args.show: imshow_torch(torch.from_numpy(P), torch.from_numpy(M), title='adv-patch')
      
      atk_flag = torch.zeros_like(Y).bool()
      for i in range(args.trial):
        AX_np = atk.apply_patch(X_np, args.scale)
        AX = torch.from_numpy(AX_np).to(device)
        if args.show: imshow_torch(X, AX, title='atk')

        if sac:
          # `raw_mask` is a predict irregular shape, `mask` is a round-corner rectangle when `shape_completion=True`; values are binarized to {0, 1}
          # `bpda` enables gradient bypassing
          RX, mask_list, raw_mask_list = sac(AX, bpda=True, shape_completion=args.sac_complete)
          RX = torch.stack(RX, dim=0)
          if args.show: imshow_torch(AX, RX, title='sac')
          AX = RX

        if mae:
          if sac:
            masks = torch.cat(mask_list, dim=0)
            pmasks = nn.AvgPool2d(PATCH_SIZE, PATCH_SIZE)(masks)
            pmasks = ThresholdSTEFunction.apply(pmasks, args.sac_thresh)
            if args.show: imshow_torch(masks, pmasks, title='mask')
            RX = denormalize(mae.recover_masked(normalize(AX), pmasks))
          else:
            _, y_hat, _ = mae(normalize(AX), args.mask_ratio)
            RX = denormalize(mae.unpatchify(y_hat))
          if args.show: imshow_torch(AX, RX, title='mae')
          AX = RX

        pred = model(normalize(AX)).argmax(dim=-1)

        atk_flag = atk_flag | (pred != Y)
        if i % 100 == 0: print(f'asr: {atk_flag.sum()/ len(atk_flag)}')
        if atk_flag.all(): break
      correct += (~atk_flag).sum()

    else:   # no atk
      pred = model(normalize(X)).argmax(dim=-1)
      correct += (pred == Y).sum()

    total += len(pred)

    if args.limit > 0 and total >= args.limit: break

  return (correct / total).item()


def go(args):
  model = get_model(args.model).to(device)
  dataloader = get_dataloader(args.batch_size)

  global sac, mae
  sac = get_sac() if args.sac else None
  mae = get_mae() if args.mae else None
  atk = get_atk(model, args.sac, args.mae) if args.atk else None

  atk_acc = test(model, dataloader, atk=atk, sac=sac, mae=mae)
  print(f'Accuracy: {atk_acc:.3%}')


if __name__ == '__main__':
  parser = ArgumentParser()
  # model & data
  parser.add_argument('-M', '--model',      default='resnet50', choices=MODELS, help='model to attack')
  parser.add_argument('-B', '--batch_size', default=16, type=int, help='batch size')
  parser.add_argument('-L', '--limit',      default=-1, type=int, help='limit run sample count')
  # adv-patch
  parser.add_argument('--atk',   action='store_true', help='enable adv-patch attack')
  parser.add_argument('--pgd',   action='store_true', help='attack method, pgd or simple Adam')
  parser.add_argument('--shape', default='circle', choices=['circle', 'square'], help='attack patch shape')
  parser.add_argument('--scale', default=0.3, type=float, help='attack patch size ratio')
  parser.add_argument('--trial', default=500, type=int,   help='attack query time limit')
  # sac
  parser.add_argument('--sac',          action='store_true', help='enable SAC defense')
  parser.add_argument('--sac_complete', action='store_true', help='enable shape completion')
  # mae
  parser.add_argument('--mae',          action='store_true',                 help='enable MAE defense')
  parser.add_argument('--mask_ratio',   default=0.3,             type=float, help='if no sac, random mask ratio')
  parser.add_argument('--sac_thresh',   default=8/PATCH_SIZE**2, type=float, help='if with sac, patch size thresh')
  # debug
  parser.add_argument('--show', action='store_true', help='shwo debug plots')
  args = parser.parse_args()

  go(args)
