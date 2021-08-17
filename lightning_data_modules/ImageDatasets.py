import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision  import transforms, datasets
import PIL.Image as Image


class ImageDataset(Dataset):
    def __init__(self, path, resolution, crop=False):
        if crop:
            crop_size = 108
            offset_height = (218 - crop_size) // 2
            offset_width = (178 - crop_size) // 2
            croper = lambda x: x[:, offset_height:offset_height + crop_size, offset_width:offset_width + crop_size]

            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Lambda(croper),
                transforms.ToPILImage(),
                transforms.Resize(size=(resolution, resolution),  interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
        else:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Resize(size=(resolution, resolution)),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)])
            
        self.dataset = datasets.ImageFolder(path, transform=transform)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)[0]

    def __len__(self):
        return self.dataset.__len__()


class CelebA(ImageDataset):
    def __init__(self, path='/store/CIA/js2164/data/celeba', resolution=64):
        ImageDataset.__init__(self, path=path, resolution=resolution)


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, config, path):
        
        self.path = path
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
        data = ImageDataset(self.path, self.resolution)
        l=len(data)
        self.train_data, self.valid_data, self.test_data = random_split(data, [int(self.split[0]*l), int(self.split[1]*l), l - int(self.split[0]*l) - int(self.split[1]*l)]) 
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size = self.train_batch, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size = self.val_batch, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_data, batch_size = self.test_batch, num_workers=self.test_workers) 