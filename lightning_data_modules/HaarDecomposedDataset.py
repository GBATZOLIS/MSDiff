import torch.utils.data as data
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from . import utils
import glob
import os
from PIL import Image

class HaarDecomposedDataset(data.Dataset):
  def __init__(self,  config, phase='train'):
    self.dataset = config.data.dataset
    self.level = config.data.level #target resolution - level 0.
    if config.data.level == 0: #data are saved as png files.
      self.image_files = glob.glob(os.path.join(config.data.base_dir, config.data.dataset, str(config.data.image_size), phase, '*.png'))
    elif config.data.level >= 1: #data are saved as numpy arrays to minimise the reconstruction.
      self.image_files = glob.glob(os.path.join(config.data.base_dir, config.data.dataset, str(config.data.image_size), phase, '*.npy'))
    else:
      raise Exception('Invalid haar level.')

    print(self.image_files)
    
    #preprocessing operations
    self.random_flip = config.data.random_flip
  
  def __getitem__(self, index):
    if self.level==0:
      image = Image.open(self.image_files[index])
      image = torch.from_numpy(np.array(image)).float()
      image = image.permute(2, 0, 1)
      image /= 255
      return image
    else:
      image = np.load(self.image_files[index])
      image = torch.from_numpy(image).float()
      return image
        
  def __len__(self):
      """Return the total number of images."""
      return len(self.image_files)

