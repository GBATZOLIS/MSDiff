from torch.utils.data import  Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
from PIL import Image, ImageOps
import torch
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from . import utils
import pytorch_lightning as pl
from glob import glob 

def listdir_nothidden_filenames(path, filetype=None):
    if not filetype:
        paths = glob(os.path.join(path, '*'))
    else:
        paths = glob(os.path.join(path, '*.%s' % filetype))
    files = [os.path.basename(path) for path in paths]
    return files

def load_data(path):
    IDs = listdir_nothidden_filenames(path)
    data = {}
    for i, ID in enumerate(IDs):
        ID_data = {}
        for quantity in listdir_nothidden_filenames(os.path.join(path, ID)):
            ID_data[quantity.split('.')[0]] = np.load(os.path.join(path, ID, quantity))
        data[i] = ID_data
    
    return data

class DUALGLOW_Dataset(Dataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self,  config, phase):
        # get the image paths of your dataset;
        self.data = load_data(os.path.join(config.data.base_dir, config.data.dataset, phase))
        print('Datapoints: %d' % len(self.data))

    def __getitem__(self, index):
        mri = self.data[index]['img_mri']
        mri = torch.tensor(mri, dtype=torch.float32).unsqueeze(0)

        pet = self.data[index]['img_pet']
        pet = torch.tensor(pet, dtype=torch.float32).unsqueeze(0)

        

        return mri, pet
        
    def __len__(self):
        """Return the total number of images."""
        return len(self.data.keys())

@utils.register_lightning_datamodule(name='DUAL-GLOW')
class DUALGLOWDataModule(pl.LightningDataModule):
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
        self.train_dataset = DUALGLOW_Dataset(self.config, phase='train')
        self.val_dataset = DUALGLOW_Dataset(self.config, phase='validation')
        self.test_dataset = DUALGLOW_Dataset(self.config, phase='validation')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.train_batch, shuffle=True, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.val_batch, shuffle=False, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size = self.test_batch, shuffle=False, num_workers=self.test_workers) 

