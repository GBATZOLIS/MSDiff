from . import utils, PairedDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

@utils.register_lightning_datamodule(name='paired')
class PairedDataModule(pl.LightningDataModule):
    def __init__(self, config):
        #DataLoader arguments
        self.config = config
        self.train_workers = config.training.workers
        self.val_workers = config.validation.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.validation.batch_size
        self.test_batch = config.eval.batch_size

    def setup(self, stage=None): 
        self.train_dataset = PairedDataset.PairedDataset(self.config, phase='train')
        self.val_dataset = PairedDataset.PairedDataset(self.config, phase='val')
        self.test_dataset = PairedDataset.PairedDataset(self.config, phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.train_batch, shuffle=True, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.val_batch, shuffle=False, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size = self.test_batch, shuffle=False, num_workers=self.test_workers) 
