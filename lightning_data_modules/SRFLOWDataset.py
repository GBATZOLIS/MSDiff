import os
# import subprocess
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import time
import torch

import pickle
import pytorch_lightning as pl
from . import utils

def get_exact_paths(config, phase):
    if config.data.dataset == 'DF2K':  
        if phase == 'train':
            LQ_file = 'DF2K-tr_X4.pklv4'
            GT_file = 'DF2K-tr.pklv4'
        elif phase == 'val':
            LQ_file = 'DIV2K-va_X4.pklv4'
            GT_file = 'DIV2K-va.pklv4'
        elif phase == 'test':
            LQ_file = 'DIV2K-teFullMod8_X4.pklv4'
            GT_file = 'DIV2K-teFullMod8.pklv4'
        else:
            return NotImplementedError('%s is not supported.' % phase)
    else:
        return NotImplementedError('%s is not supported.' % config.data.dataset)
    
    full_path_LQ = os.path.join(config.data.base_dir, config.data.dataset, LQ_file)
    full_path_GT = os.path.join(config.data.base_dir, config.data.dataset, GT_file)

    return {'LQ':full_path_LQ, 'GT':full_path_GT}

class LRHR_PKLDataset(data.Dataset):
    def __init__(self, config, phase):
        super(LRHR_PKLDataset, self).__init__()
        self.crop_size = config.data.target_resolution
        self.scale = None
        self.random_scale_list = [1]

        hr_file_path = get_exact_paths(config, phase)['GT']
        lr_file_path = get_exact_paths(config, phase)['LQ']

        self.use_flip = config.data.use_flip
        self.use_rot = config.data.use_rot
        self.use_crop = config.data.use_crop
        #self.center_crop_hr_size = opt.get("center_crop_hr_size", None)

        #n_max = opt["n_max"] if "n_max" in opt.keys() else int(1e8)

        t = time.time()
        self.lr_images = self.load_pkls(lr_file_path, n_max=int(1e9))
        self.hr_images = self.load_pkls(hr_file_path, n_max=int(1e9))

        min_val_hr = np.min([i.min() for i in self.hr_images[:20]])
        max_val_hr = np.max([i.max() for i in self.hr_images[:20]])

        min_val_lr = np.min([i.min() for i in self.lr_images[:20]])
        max_val_lr = np.max([i.max() for i in self.lr_images[:20]])

        t = time.time() - t
        print("Loaded {} HR images with [{:.2f}, {:.2f}] in {:.2f}s from {}".
              format(len(self.hr_images), min_val_hr, max_val_hr, t, hr_file_path))
        print("Loaded {} LR images with [{:.2f}, {:.2f}] in {:.2f}s from {}".
              format(len(self.lr_images), min_val_lr, max_val_lr, t, lr_file_path))

    def load_pkls(self, path, n_max):
        assert os.path.isfile(path), path
        images = []
        with open(path, "rb") as f:
            images += pickle.load(f)
        assert len(images) > 0, path
        images = images[:n_max]
        images = [np.transpose(image, [2, 0, 1]) for image in images]
        return images

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, item):

        hr = self.hr_images[item]
        lr = self.lr_images[item]

        if self.scale == None:
            self.scale = hr.shape[1] // lr.shape[1]
            assert hr.shape[1] == self.scale * lr.shape[1], ('non-fractional ratio', lr.shape, hr.shape)

        if self.use_crop:
            hr, lr = random_crop(hr, lr, self.crop_size, self.scale, self.use_crop)

        #if self.center_crop_hr_size:
        #    hr, lr = center_crop(hr, self.center_crop_hr_size), center_crop(lr, self.center_crop_hr_size // self.scale)

        if self.use_flip:
            hr, lr = random_flip(hr, lr)

        if self.use_rot:
            hr, lr = random_rotation(hr, lr)

        hr = hr / 255.0
        lr = lr / 255.0

        hr = torch.Tensor(hr)
        lr = torch.Tensor(lr)

        return lr, hr

def random_flip(img, seg):
    random_choice = np.random.choice([True, False])
    img = img if random_choice else np.flip(img, 2).copy()
    seg = seg if random_choice else np.flip(seg, 2).copy()
    return img, seg


def random_rotation(img, seg):
    random_choice = np.random.choice([0, 1, 3])
    img = np.rot90(img, random_choice, axes=(1, 2)).copy()
    seg = np.rot90(seg, random_choice, axes=(1, 2)).copy()
    return img, seg


def random_crop(hr, lr, size_hr, scale, random):
    size_lr = size_hr // scale

    size_lr_x = lr.shape[1]
    size_lr_y = lr.shape[2]

    start_x_lr = np.random.randint(low=0, high=(size_lr_x - size_lr) + 1) if size_lr_x > size_lr else 0
    start_y_lr = np.random.randint(low=0, high=(size_lr_y - size_lr) + 1) if size_lr_y > size_lr else 0

    # LR Patch
    lr_patch = lr[:, start_x_lr:start_x_lr + size_lr, start_y_lr:start_y_lr + size_lr]

    # HR Patch
    start_x_hr = start_x_lr * scale
    start_y_hr = start_y_lr * scale
    hr_patch = hr[:, start_x_hr:start_x_hr + size_hr, start_y_hr:start_y_hr + size_hr]

    return hr_patch, lr_patch


def center_crop(img, size):
    assert img.shape[1] == img.shape[2], img.shape
    border_double = img.shape[1] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, border:-border, border:-border]


def center_crop_tensor(img, size):
    assert img.shape[2] == img.shape[3], img.shape
    border_double = img.shape[2] - size
    assert border_double % 2 == 0, (img.shape, size)
    border = border_double // 2
    return img[:, :, border:-border, border:-border]


@utils.register_lightning_datamodule(name='LRHR_PKLDataset')
class PairedDataModule(pl.LightningDataModule):
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
        self.train_dataset = LRHR_PKLDataset(self.config, phase='train')
        self.val_dataset = LRHR_PKLDataset(self.config, phase='val')
        self.test_dataset = LRHR_PKLDataset(self.config, phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.train_batch, shuffle=True, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.val_batch, shuffle=False, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size = self.test_batch, shuffle=False, num_workers=self.test_workers) 

