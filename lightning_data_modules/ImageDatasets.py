import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision  import transforms, datasets
import PIL.Image as Image
from . import utils

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




