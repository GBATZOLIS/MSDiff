import torch.utils.data as data
from torchvision import transforms
from PIL import Image, ImageOps
import torch
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm




class PairedDataset(data.Dataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self,  config, phase, domain=None):
        #set domain to the domain you want to sample from. If domain is set to N we get paired samples from all domains.
        self.domain = domain

        # get the image paths of your dataset;
        self.image_paths = load_image_paths(os.path.join(config.data.base_dir, config.data.dataset), phase)
        _, file_extension = os.path.splitext(self.image_paths['A'][0])
        self.file_extension = file_extension

        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        if self.file_extension in ['.jpg', '.png']:
            transform_list = [transforms.ToTensor()]
        elif self.file_extension in ['.npy']:
            self.channels = config.data.num_channels
            self.resolution = config.data.image_size
            transform_list = [torch.from_numpy, lambda x: x.type(torch.FloatTensor)]
        else:
            raise Exception('File extension %s is not supported yet. Please update the code.' % self.file_extension)

        self.transform = transforms.Compose(transform_list)


    def __getitem__(self, index):
        if self.domain is None:
            A_path = self.image_paths['A'][index]
            B_path = self.image_paths['B'][index]

            #load the paired images/scans
            if self.file_extension in ['.jpg', '.png']:
                A = Image.open(A_path).convert('RGB')
                B = Image.open(B_path).convert('RGB')
            elif self.file_extension in ['.npy']:
                A = np.load(A_path)
                B = np.load(B_path)

                #reshape/slice appropriately
                if self.channels == 1:
                    #slicing
                    def get_starting_index(A, resolution, axis):
                        if A.shape[axis] == self.resolution[axis]:
                            starting_index = 0
                        elif A.shape[axis] > self.resolution[axis]:
                            starting_index = np.random.randint(0, A.shape[axis]-self.resolution[axis])
                        else:
                            raise Exception('requested resolution exceeds data resolution in axis %d' % axis)
                        return starting_index

                    #i0, i1, i2 = get_starting_index(A, self.resolution, 0), get_starting_index(A, self.resolution, 1), get_starting_index(A, self.resolution, 2)
                    #A = A[i0:i0+self.resolution[0], i1:i1+self.resolution[1], i2:i2+self.resolution[2]]
                    #B = B[i0:i0+self.resolution[0], i1:i1+self.resolution[1], i2:i2+self.resolution[2]]

                    #------rotation-------
                    #angle = [0, 90, 180, 270][np.random.randint(4)]
                    #axes_combo = [(0, 1), (1, 2), (0, 2)][np.random.randint(3)]
                    #if angle != 0:
                    #    A = scipy.ndimage.rotate(A, angle=angle, axes=axes_combo)
                    #    B = scipy.ndimage.rotate(B, angle=angle, axes=axes_combo)

                    #dequantise 0 value
                    #A[A==0.]=10**(-6)*np.random.rand()
                    #B[B==0.]=10**(-6)*np.random.rand()
                    
                    #expand dimensions to acquire a pytorch-like form.
                    A = np.expand_dims(A, axis=0)
                    B = np.expand_dims(B, axis=0)

                elif self.channels > 1 and self.channels < A.shape[-1]:
                    starting_slicing_index = np.random.randint(0, A.shape[-1] - self.channels)
                    A = A[:,:,starting_slicing_index:starting_slicing_index+self.channels]
                    A = np.moveaxis(A, -1, 0)
                    B = B[:,:,starting_slicing_index:starting_slicing_index+self.channels]
                    B = np.moveaxis(B, -1, 0)

                elif self.channels == A.shape[-1]:
                    A = np.moveaxis(A, -1, 0)
                    B = np.moveaxis(B, -1, 0)
                else:
                    raise Exception('Invalid number of channels.')

            else:
                raise Exception('File extension %s is not supported yet. Please update the code.' % self.file_extension)
            
            #transform the images/scans
            A_transformed = self.transform(A)
            B_transformed = self.transform(B)

            #print(A_transformed.min(), A_transformed.max())
            #print(B_transformed.min(), B_transformed.max())

            return A_transformed, B_transformed
        else:
            path = self.image_paths[self.domain][index]

            #load the image/scan
            if self.file_extension in ['.jpg', '.png']:
                img = Image.open(path).convert('RGB')
            elif self.file_extension in ['.npy']:
                img = np.load(path)

                #dequantise 0 value
                #img[img<10**(-6)]=10**(-6)*np.random.rand()

                #reshape/slice appropriately
                if self.channels == 1:
                    img = np.expand_dims(img, axis=0)
                elif self.channels > 1 and self.channels < img.shape[-1]:
                    starting_slicing_index = np.random.randint(0, img.shape[-1] - self.channels)
                    img = img[:,:,starting_slicing_index:starting_slicing_index+self.channels]
                    img = np.moveaxis(img, -1, 0)
                elif self.channels == img.shape[-1]:
                    img = np.moveaxis(img, -1, 0)
                else:
                    raise Exception('Invalid number of channels.')

            #transform the image/scan
            img_transformed = self.transform(img)
            
            return img_transformed
        
    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths['A'])



def center_crop(img, crop_left, crop_right, crop_top, crop_bottom):
    width, height = img.size
    left = crop_left
    right = width - crop_right
    top = crop_top
    bottom = height - crop_bottom
    return img.crop((left, top, right, bottom))

def create_train_val_test_index_dict(total_num_images, split):
    #return a dictionary that maps each index to the corresponding phase dataset (train, val, test)
    indices = np.arange(total_num_images)
    np.random.shuffle(indices) #in-place operation
    phase_dataset = {}
    for counter, index in enumerate(indices):
        if counter < split[0]*total_num_images:
            folder = 'train'
        elif counter < (split[0]+split[1])*total_num_images:
            folder = 'val'
        else:
            folder = 'test'
        phase_dataset[index] = folder
    return phase_dataset

def create_dataset(master_path='caflow/datasets/edges2shoes', resize_size=32, dataset_size=2000, dataset_style='image2image', split=[0.8,0.1,0.1]):
    data_paths = make_dataset(master_path)

    Path(os.path.join(master_path, 'train')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(master_path, 'val')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(master_path, 'test')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(master_path, 'train','A')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(master_path, 'val', 'A')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(master_path, 'test', 'A')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(master_path, 'train', 'B')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(master_path, 'val'), 'B').mkdir(parents=True, exist_ok=True)
    Path(os.path.join(master_path, 'test'), 'B').mkdir(parents=True, exist_ok=True)

    phase_dataset = create_train_val_test_index_dict(len(data_paths), split)
    for counter, AB_path in tqdm(enumerate(sorted(data_paths))):
        basename = os.path.basename(AB_path)
        AB = Image.open(AB_path).convert('RGB')

        #different datasets require different processing which is implemented here for each different dataset.
        if dataset_style == 'image2image':
            # crop
            w, h = AB.size
            w2 = int(w / 2)
            A = AB.crop((0, 0, w2, h))
            B = AB.crop((w2, 0, w, h))
            # resize
            if isinstance(resize_size, int):
                resize_size = (resize_size, resize_size)
            A_resize = A.resize(resize_size, Image.BICUBIC)
            B_resize = B.resize(resize_size, Image.BICUBIC)
        
        # save
        A_resize.save(os.path.join(master_path, phase_dataset[counter], 'A', basename))
        B_resize.save(os.path.join(master_path, phase_dataset[counter], 'B', basename))


def inspect_dataset(master_path, resize_size, dataset_size):
    info = {'train':{'A':{'count':0, 'names':[], 'size':None}, \
                     'B':{'count':0, 'names':[], 'size':None}}, \
            'val':{'A':{'count':0, 'names':[], 'size':None}, \
                     'B':{'count':0, 'names':[], 'size':None}}, \
            'test':{'A':{'count':0, 'names':[], 'size':None}, \
                     'B':{'count':0, 'names':[], 'size':None}}}
    
    for phase in ['train', 'val', 'test']:
        for domain in ['A', 'B']:
            subpath=os.path.join(master_path, phase, domain)
            if not os.path.exists(subpath):
                   os.mkdir(subpath)
            else:
                i=0
                for root, _, fnames in os.walk(subpath):
                    phase_domain_names = []
                    for fname in fnames:    
                        if is_image_file(fname):
                            info[phase][domain]['count'] += 1
                            phase_domain_names.append(os.path.basename(fname))
                            if i==0:
                                img = Image.open(os.path.join(subpath, fname)).convert('RGB')
                                w, h = img.size
                                info[phase][domain]['size'] = w #we assume w = h
                            i+=1
                    
                    info[phase][domain]['names'] = sorted(phase_domain_names)

    empty = {'train':True, 'val':True, 'test':True}
    for phase in ['train', 'val', 'test']:
        if info[phase]['A']['count']>0 or info[phase]['B']['count']>0:
            empty[phase]=False

    for phase in ['train', 'val', 'test']:
        try:
            #check the count
            assert info[phase]['A']['count'] == info[phase]['B']['count'], \
            'Different count number between A and B domains.'

            if phase == 'train':
                assert info[phase]['A']['count'] == dataset_size, 'Dataset size different than requested.'

            #check image size
            assert info[phase]['A']['size'] == resize_size, 'Domain A has different size than size requested'
            assert info[phase]['B']['size'] == resize_size, 'Domain B has different size than size requested'

            #check image pairing
            for i in range(info[phase]['A']['count']):
                assert info[phase]['A']['names'][i]==info[phase]['B']['names'][i], \
                'Wrong image pairing. A:{} - B:{}'.format(info[phase]['A']['names'][i], info[phase]['B']['names'][i])

        except AssertionError:
            for domain in ['A', 'B']:
                subpath = os.path.join(master_path, phase, domain)
                for root, _, fnames in os.walk(subpath):
                    for fname in fnames:
                        os.remove(os.path.join(subpath, fname))
                
            empty[phase]=True
        
    datasets_to_create = []
    for phase in ['train', 'val', 'test']:
        if empty[phase] == True:
            datasets_to_create.append(phase)
        
    return datasets_to_create

def load_image_paths(master_path, phase):
    assert os.path.isdir(os.path.join(master_path, phase)), '%s is not a valid directory' % dir
    #print('load img dir: {}'.format(os.path.join(master_path, phase)))
    domains = ['A', 'B']
    images = {}
    for domain in domains:
        for root, _, fnames in os.walk(os.path.join(master_path, phase, domain)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    if os.path.basename(path) not in images.keys():
                        images[os.path.basename(path)]=[]
                        images[os.path.basename(path)].append(path)
                    else:
                        images[os.path.basename(path)].append(path)
    
    #for key in list(images.keys())[:10]:
    #    print('{} - {} - {}'.format(key, images[key][0], images[key][1]))

    load_images = {'A':[], 'B':[]}
    for key in images.keys():
        load_images['A'].append(images[key][0])
        load_images['B'].append(images[key][1])

    print(load_images['A'][:3])
    print(load_images['B'][:3])

    #assertions
    assert len(load_images['A'])==len(load_images['B']), 'There is a mismatch in the number of domain A and domain B images.'
    for i in range(len(load_images['A'])):
        assert os.path.basename(load_images['A'][i])==os.path.basename(load_images['B'][i]), \
               'The images are not paired. A:{} - B:{}'.format(os.path.basename(load_images['A'][i]), os.path.basename(load_images['B'][i]))

    return load_images


def is_image_file(filename):
    IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.webp', '.npy'
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
                
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')