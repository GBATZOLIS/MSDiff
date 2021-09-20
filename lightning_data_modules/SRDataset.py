import torch.utils.data as data
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from . import utils
import glob
import os
from PIL import Image
import bicubic_pytorch.imresize as imresize
from torchvision.transforms import RandomCrop, CenterCrop, ToTensor, Resize, InterpolationMode
import random

def get_img_paths(paths, phase):
    if phase == 'train':
        return paths[:162770]
    elif phase == 'val':
        return random.shuffle(paths[162770:182637])[:5000]
    else:
        return random.shuffle(paths[182637:])[:5000]

class SuperResolutionDataset(data.Dataset):
    def __init__(self,  config, phase='train'):
        self.dataset = config.data.dataset
        self.level = config.data.level
        
        paths = sorted(glob.glob(os.path.join(config.data.base_dir, config.data.dataset,'*.jpg'))) 
        print(paths[:5])
        self.image_files = get_img_paths(paths, phase)
        
        self.convert_to_tensor = ToTensor()

        if phase == 'train':
            self.crop_to_GT_size = RandomCrop(size=config.data.target_resolution)
        else:
            self.crop_to_GT_size = CenterCrop(size=config.data.target_resolution)

        self.resize_to_hr = Resize(config.data.image_size//2**config.data.level, interpolation=InterpolationMode.BICUBIC)
        self.resize_to_lr = Resize(config.data.image_size//2**(config.data.level+1), interpolation=InterpolationMode.BICUBIC)


    def __get_item(self, index):
        image = self.convert_to_tensor(Image.open(self.image_files[index]).convert('RGB'))
        print(image.size())

        cropped_image = self.crop_to_GT_size(image)
        print(cropped_image.size())

        hr = self.resize_to_hr(cropped_image)
        print(hr.size())

        lr = self.resize_to_lr(cropped_image)
        print(lr.size())

        return lr, hr 

@utils.register_lightning_datamodule(name='bicubic_multiscale')
class SuperResolutionDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        #DataLoader arguments
        self.config = config
        self.train_workers = config.training.workers
        self.val_workers = config.eval.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.eval.batch_size
        self.test_batch = config.eval.batch_size

    def setup(self, stage=None): 
        self.train_dataset = SuperResolutionDataset(self.config, phase='train')
        self.val_dataset = SuperResolutionDataset(self.config, phase='val')
        self.test_dataset = SuperResolutionDataset(self.config, phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.train_batch, shuffle=True, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.val_batch, shuffle=False, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size = self.test_batch, shuffle=False, num_workers=self.test_workers) 