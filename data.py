#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import os
import json
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

from utils import *


DATASET_STATS = {
  'imagenet': (
    (0.485, 0.456, 0.406),  # mean
    (0.229, 0.224, 0.225),  # var
  ),
  'cifar10': (
    (0.4914, 0.4822, 0.4465),
    (0.2471, 0.2435, 0.2616),
  ),
}
DATASETS = list(DATASET_STATS.keys())

class ImageNet_1k(Dataset):

  def __init__(self, root: str, transform=None):
    self.base_path = os.path.join(root, 'val')
    self.transform = transform

    fns = sorted([fn for fn in os.listdir(self.base_path)])
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


def get_dataloader(name:str, split:str='test', batch_size:int=32) -> DataLoader:
  assert split in ['train', 'val', 'test']
  is_train = split == 'train'
  
  if name == 'imagenet':
    assert split == 'test', 'imagenet-k only has test set'
    dataset = ImageNet_1k(root='data/imagenet-1k', transform=TF.to_tensor)
  elif name == 'cifar10':
    dataset = CIFAR10(root='data', train=is_train, transform=TF.to_tensor, download=True)
  return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, pin_memory=False, num_workers=0)


DATASET_STATS_CACHE = { }

@torch.inference_mode(False)
def _make_stats_cache(dataset:str):
  avg, std = DATASET_STATS[dataset]
  avg = Tensor(list(avg)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
  std = Tensor(list(std)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)
  DATASET_STATS_CACHE[dataset] = avg, std

def normalize(dataset:str, X:Tensor) -> Tensor:
  if dataset not in DATASET_STATS_CACHE: _make_stats_cache(dataset)
  avg, std = DATASET_STATS_CACHE[dataset]
  return TF.normalize(X, avg, std)       # [B, C, H, W]

def denormalize(dataset:str, X:Tensor) -> Tensor:
  if dataset not in DATASET_STATS_CACHE: _make_stats_cache(dataset)
  avg, std = DATASET_STATS_CACHE[dataset]
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
