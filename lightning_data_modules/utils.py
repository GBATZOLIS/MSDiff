from . import PairedDataset, HaarDecomposedDataset, ImageDatasets, SyntheticDataset
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split


_LIGHTNING_DATA_MODULES = {}
def register_lightning_datamodule(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _LIGHTNING_DATA_MODULES:
      raise ValueError(f'Already registered model with name: {local_name}')
    _LIGHTNING_DATA_MODULES[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_lightning_datamodule_by_name(name):
  print(_LIGHTNING_DATA_MODULES.keys())
  return _LIGHTNING_DATA_MODULES[name]

def create_lightning_datamodule(config):
  datamodule = get_lightning_datamodule_by_name(config.data.datamodule)(config)
  return datamodule


@register_lightning_datamodule(name='paired')
class PairedDataModule(pl.LightningDataModule):
    def __init__(self, config):
        #DataLoader arguments
        self.config = config
        self.train_workers = config.training.workers
        self.val_workers = config.eval.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.eval.batch_size
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

@register_lightning_datamodule(name='haar_multiscale')
class HaarDecomposedDataModule(pl.LightningDataModule):
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
        self.train_dataset = HaarDecomposedDataset.HaarDecomposedDataset(self.config, phase='train')
        self.val_dataset = HaarDecomposedDataset.HaarDecomposedDataset(self.config, phase='val')
        self.test_dataset = HaarDecomposedDataset.HaarDecomposedDataset(self.config, phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.train_batch, shuffle=True, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.val_batch, shuffle=False, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size = self.test_batch, shuffle=False, num_workers=self.test_workers) 

@register_lightning_datamodule(name='image')
class ImageDataModule(pl.LightningDataModule):
    def __init__(self, config):
        self.path = config.data.base_dir
        self.resolution = config.data.image_size
        self.split = config.data.split

        #DataLoader arguments
        self.train_workers = config.training.workers
        self.val_workers = config.validation.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.validation.batch_size
        self.test_batch = config.eval.batch_size

    def setup(self, stage=None): 
        data = ImageDatasets.ImageDataset(self.path, self.resolution)
        l=len(data)
        self.train_data, self.valid_data, self.test_data = random_split(data, [int(self.split[0]*l), int(self.split[1]*l), l - int(self.split[0]*l) - int(self.split[1]*l)]) 
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers) 

@register_lightning_datamodule(name='Synthetic')
class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, config): 
        super(SyntheticDataModule, self).__init__()
        #Synthetic Dataset arguments
        self.data_samples=config.data.data_samples
        self.dataset_type=config.data.dataset_type
        self.mixtures = config.data.mixtures
        self.return_mixtures = config.data.return_mixtures
        self.split = config.data.split

        #DataLoader arguments
        self.train_workers = config.training.workers
        self.val_workers = config.validation.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.validation.batch_size
        self.test_batch = config.eval.batch_size

        #self.normalize = config.normalize
        
    def setup(self, stage=None): 
        data = SyntheticDataset.SyntheticDataset(self.data_samples, self.dataset_type, self.mixtures, self.return_mixtures)
        l=len(data)
        self.train_data, self.valid_data, self.test_data = random_split(data, [int(self.split[0]*l), int(self.split[1]*l), int(self.split[2]*l)]) 
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers) 
    