from lightning_modules.utils import create_lightning_module
from lightning_data_modules.utils import create_lightning_datamodule
from tqdm import tqdm
import os
import torch 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def max_pairwise_L2_distance(batch):
    num_images = batch.size(0)
    max_distance = float("-inf")
    for i in range(num_images):
      for j in range(i+1, num_images):
        distance = torch.norm(batch[i]-batch[j], p=2)
        if distance > max_distance:
          max_distance = distance
    return max_distance


def compute_dataset_statistics(config):
  if config.data.dataset=='celebA':
    mean_save_dir = os.path.join(config.data.base_dir, 'datasets_mean', config.data.dataset+'_'+str(config.data.image_size))
    Path(mean_save_dir).mkdir(parents=True, exist_ok=True)

    config.training.batch_size = 128
    DataModule = create_lightning_datamodule(config)
    DataModule.setup()
    train_dataloader = DataModule.train_dataloader()

    LightningModule = create_lightning_module(config).to('cuda:0')


    with torch.no_grad():
      total_sum = None
      total_num_images = 0
      max_val = float('-inf')
      min_val = float('inf')
      #max_distance = float('-inf')
      for i, batch in tqdm(enumerate(train_dataloader)):
          hf = LightningModule.get_hf_coefficients(batch.to('cuda:0'))
          
          #calculate max pairwise distance
          #max_batch_pairwise_distance = max_pairwise_L2_distance(hf)
          #if max_batch_pairwise_distance > max_distance:
          #  max_distance = max_batch_pairwise_distance

          if hf.min() < min_val:
            min_val = hf.min()
          if hf.max() > max_val:
            max_val = hf.max()

          num_images = hf.size(0)
          total_num_images += num_images
          batch_sum = torch.sum(hf, dim=0)

          if total_sum is None:
            total_sum = batch_sum
          else:
            total_sum += batch_sum
    
    #print('Max pairwise distance: %.4f' % max_distance)

    print('range: [%.5f, %.5f]' % (min_val, max_val))
    print('total_num_images: %d' % total_num_images)
    mean = total_sum / total_num_images
    mean = mean.cpu()
    print(mean.size())
    
    torch.save(mean, f=os.path.join(mean_save_dir, 'mean.pt'))

    mean = mean.numpy().flatten()

    print('Maximum mean value: ', np.amax(mean))
    print('Minimum mean value: ', np.amin(mean))

    plt.figure()
    plt.title('Mean values histogram')
    _ = plt.hist(mean, bins='auto')
    plt.savefig(os.path.join(mean_save_dir, 'mean_histogram.png'))
  
  elif config.data.dataset == 'mri_to_pet':
    dataset_info_dir = os.path.join(config.data.base_dir, 'datasets_info', config.data.dataset)
    Path(dataset_info_dir).mkdir(parents=True, exist_ok=True)
    DataModule = create_lightning_datamodule(config)
    DataModule.setup()
    train_dataloader = DataModule.train_dataloader()

    info = {'min_vals':{'mri':[], 'pet':[]}, 'max_vals':{'mri':[], 'pet':[]}, 'ranges':{'mri':[], 'pet':[]}}

    for i, batch in tqdm(enumerate(train_dataloader)):
      mri, pet = batch
      mri_min, mri_max = torch.min(mri).item(), torch.max(mri).item()
      pet_min, pet_max = torch.min(pet).item(), torch.max(pet).item()

      info['min_vals']['mri'].append(mri_min)
      info['min_vals']['pet'].append(pet_min)

      info['max_vals']['mri'].append(mri_max)
      info['max_vals']['pet'].append(pet_max)

      info['ranges']['mri'].append(mri_max-mri_min)
      info['ranges']['pet'].append(pet_max-pet_min)
    
    for quantity in info.keys():
      for modality in quantity.keys():
        plt.figure()
        plt.title('%s %s histogram' % (modality, quantity))
        _ = plt.hist(info[quantity][modality], bins='auto')
        plt.savefig(os.path.join(dataset_info_dir, 'mean_histogram.png'))





