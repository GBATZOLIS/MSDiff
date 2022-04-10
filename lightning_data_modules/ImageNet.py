import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision  import transforms, datasets
import PIL.Image as Image
from . import utils
import os
import glob

def load_file_paths(dataset_base_dir, phase):
    if phase in ['val', 'test']:
        listOfFiles = [os.path.join(dataset_base_dir, phase, f) for f in os.listdir(dataset_base_dir, phase)]
    elif phase == 'train':
        listOfFiles = []
        for class_folder in os.listdir(os.path.join(dataset_base_dir, phase)):
            class_dir = os.path.join(os.path.join(dataset_base_dir, phase, class_folder))
            for class_image in os.listdir(class_dir):
                listOfFiles.append(os.path.join(class_dir, class_image))

    return listOfFiles

#the code should become more general for the ImageDataset class.
class ImageNetDataset(Dataset):
    def __init__(self, config, phase):
        path = os.path.join(config.data.base_dir, config.data.dataset)
        res_x, res_y = config.data.shape[1], config.data.shape[2]
        self.transform = transforms.Compose(
                        [transforms.ToTensor(),
                        transforms.Resize(size=(res_x, res_y))])
            
        self.image_paths = load_file_paths(path, phase)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert('RGB')
        image = self.transform(image)
        print(image)
        return image

    def __len__(self):
        return len(self.image_paths)


@utils.register_lightning_datamodule(name='ImageNet')
class ImageDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        #DataLoader arguments
        self.train_workers = config.training.workers
        self.val_workers = config.eval.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.eval.batch_size
        self.test_batch = config.eval.batch_size

    def setup(self, stage=None): 
        self.train_data = ImageNetDataset(self.config, 'train')
        self.valid_data = ImageNetDataset(self.config, 'val')
        self.test_data = ImageNetDataset(self.config, 'test')

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers) 
