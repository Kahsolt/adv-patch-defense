#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import os
import json
from PIL import Image
import warnings ; warnings.filterwarnings("ignore", category=UserWarning)

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.models as M
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Union
from torch import Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True

''' data '''

class ImageNet_1k(Dataset):

  def __init__(self, root: str, transform=None):
    self.base_path = os.path.join(root, 'val')
    self.transform = transform

    fns = [fn for fn in os.listdir(self.base_path)]
    fps = [os.path.join(self.base_path, fn) for fn in fns]
    with open(os.path.join(root, 'image_name_to_class_id_and_name.json'), encoding='utf-8') as fh:
      mapping = json.load(fh)
    tgts = [mapping[fn]['class_id'] for fn in fns]

    self.metadata = [x for x in zip(fps, tgts)]

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    fp, tgt = self.metadata[idx]
    img = Image.open(fp)
    img = img.convert('RGB')
    if self.transform is not None:
      img = self.transform(img)
    return img, tgt

def get_dataloader(bs=32) -> DataLoader:
  dataset = ImageNet_1k(root='data/imagenet-1k', transform=TF.to_tensor)
  return DataLoader(dataset, batch_size=bs, shuffle=False, pin_memory=False, num_workers=0)


def imagenet_stats() -> Tuple[Tuple[float], Tuple[float]]:
  avg = (0.485, 0.456, 0.406)
  std = (0.229, 0.224, 0.225)
  return avg, std

def normalize(X: torch.Tensor) -> torch.Tensor:
  ''' NOTE: to insure attack validity, normalization is delayed until enumerating a dataloader '''
  avg, std = imagenet_stats()
  return TF.normalize(X, avg, std)       # [B, C, H, W]

def denormalize(X: torch.Tensor) -> torch.Tensor:
  avg, std = imagenet_stats()
  avg = torch.Tensor(list(avg)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(X.device)
  std = torch.Tensor(list(std)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(X.device)
  return X * std + avg

def minmax_norm(X:Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
  return (X - X.min()) / (X.max() - X.min())


def imshow_torch(X:Tensor, Y:Tensor=None, title=''):
  tc2np = lambda X: X.permute([1, 2, 0]).detach().cpu().numpy()

  if Y is None:
    plt.imshow(tc2np(make_grid(X))) ; plt.axis('off')
  else:
    plt.subplot(121) ; plt.imshow(tc2np(make_grid(X))) ; plt.axis('off')
    plt.subplot(122) ; plt.imshow(tc2np(make_grid(Y))) ; plt.axis('off')
  plt.suptitle(title)
  plt.show()


''' model '''

MODELS = [
  'alexnet', 

#  'vgg11',
#  'vgg13',
#  'vgg16',
#  'vgg19',
#  'vgg11_bn',
#  'vgg13_bn',
#  'vgg16_bn',
  'vgg19_bn',

  'convnext_tiny',
#  'convnext_small',
#  'convnext_base',
#  'convnext_large',
  
  'densenet121',
#  'densenet161',
#  'densenet169',
#  'densenet201',

#  'efficientnet_b0',
#  'efficientnet_b1',
#  'efficientnet_b2',
#  'efficientnet_b3',
#  'efficientnet_b4',
#  'efficientnet_b5',
#  'efficientnet_b6',
#  'efficientnet_b7',
  'efficientnet_v2_s',
#  'efficientnet_v2_m',
#  'efficientnet_v2_l',

#  'googlenet',

  'inception_v3',

#  'mnasnet0_5',
#  'mnasnet0_75',
#  'mnasnet1_0',
  'mnasnet1_3',

#  'mobilenet_v2',
#  'mobilenet_v3_small',
  'mobilenet_v3_large',

  'regnet_y_400mf',
#  'regnet_y_800mf',
#  'regnet_y_1_6gf',
#  'regnet_y_3_2gf',
#  'regnet_y_8gf',
#  'regnet_y_16gf',
#  'regnet_y_32gf',
#  'regnet_y_128gf',
#  'regnet_x_400mf',
#  'regnet_x_800mf',
#  'regnet_x_1_6gf',
#  'regnet_x_3_2gf',
#  'regnet_x_8gf',
#  'regnet_x_16gf',
#  'regnet_x_32gf',

  'resnet18',
  'resnet34',
  'resnet50',
#  'resnet101',
#  'resnet152',
  'resnext50_32x4d',
#  'resnext101_32x8d',
#  'resnext101_64x4d',
  'wide_resnet50_2',
#  'wide_resnet101_2',

#  'shufflenet_v2_x0_5',
#  'shufflenet_v2_x1_0',
  'shufflenet_v2_x1_5',
#  'shufflenet_v2_x2_0',

#  'squeezenet1_0',
  'squeezenet1_1',

  'vit_b_16',
#  'vit_b_32',
#  'vit_l_16',
#  'vit_l_32',
#  'vit_h_14',

  'swin_t',
#  'swin_s',
#  'swin_b',
]

def get_model(name, ckpt_fp=None):
  if hasattr(M, name):
    model = getattr(M, name)(pretrained=ckpt_fp is None)
    if ckpt_fp:
      model.load_state_dict(torch.load(ckpt_fp))
  else:
    raise ValueError(f'[get_model] unknown model {name}')
  return model
