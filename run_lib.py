from models import ddpm, ncsnv2, fcn #needed for model registration
import pytorch_lightning as pl
import numpy as np

from torchvision.utils import make_grid

from lightning_callbacks import callbacks, HaarMultiScaleCallback, PairedCallback #needed for callback registration
from lightning_callbacks.HaarMultiScaleCallback import normalise_per_image, permute_channels, normalise, normalise_per_band, create_supergrid
from lightning_callbacks.utils import get_callbacks

from lightning_data_modules import HaarDecomposedDataset, ImageDatasets, PairedDataset, SyntheticDataset #needed for datamodule registration
from lightning_data_modules.utils import create_lightning_datamodule

from lightning_modules import BaseSdeGenerativeModel, HaarMultiScaleSdeGenerativeModel, ConditionalSdeGenerativeModel #need for lightning module registration
from lightning_modules.utils import create_lightning_module

import create_dataset
from torch.nn import Upsample
import torch 

from pathlib import Path
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

def train(config, log_path, checkpoint_path):
    if config.data.create_dataset:
      create_dataset.create_dataset(config)

    DataModule = create_lightning_datamodule(config)
    callbacks = get_callbacks(config)
    LightningModule = create_lightning_module(config)

    logger = pl.loggers.TensorBoardLogger(log_path, name='lightning_logs')

    if checkpoint_path is not None or config.model.checkpoint_path is not None:
      if config.model.checkpoint_path is not None and checkpoint_path is None:
        checkpoint_path = config.model.checkpoint_path

      trainer = pl.Trainer(gpus=config.training.gpus,
                          num_nodes = config.training.num_nodes,
                          accelerator = config.training.accelerator,
                          accumulate_grad_batches = config.training.accumulate_grad_batches,
                          gradient_clip_val = config.optim.grad_clip,
                          max_steps=config.training.n_iters, 
                          callbacks=callbacks, 
                          logger = logger,
                          resume_from_checkpoint=checkpoint_path)
    else:  
      trainer = pl.Trainer(gpus=config.training.gpus,
                          num_nodes = config.training.num_nodes,
                          accelerator = config.training.accelerator,
                          accumulate_grad_batches = config.training.accumulate_grad_batches,
                          gradient_clip_val = config.optim.grad_clip,
                          max_steps=config.training.n_iters,
                          callbacks=callbacks,
                          logger = logger                          
                          )

    trainer.fit(LightningModule, datamodule=DataModule)

def test(config, log_path, checkpoint_path):
  DataModule = create_lightning_datamodule(config)
  DataModule.setup() #instantiate the datasets

  callbacks = get_callbacks(config)
  LightningModule = create_lightning_module(config)
  logger = pl.loggers.TensorBoardLogger(log_path, name='test_lightning_logs')

  trainer = pl.Trainer(gpus=config.training.gpus,
                      callbacks=callbacks, 
                      logger = logger,
                      resume_from_checkpoint=checkpoint_path)

  trainer.test(LightningModule, DataModule.test_dataloader())

def compute_dataset_statistics(config):
  def max_pairwise_L2_distance(batch):
    num_images = batch.size(0)
    max_distance = float("-inf")
    for i in range(num_images):
      for j in range(i+1, num_images):
        distance = torch.norm(batch[i]-batch[j], p=2)
        if distance > max_distance:
          max_distance = distance
    return max_distance




  mean_save_dir = os.path.join(config.data.base_dir, 'datasets_mean', config.data.dataset+'_'+str(config.data.image_size))
  Path(mean_save_dir).mkdir(parents=True, exist_ok=True)

  config.training.batch_size = 1024
  DataModule = create_lightning_datamodule(config)
  DataModule.setup()
  train_dataloader = DataModule.train_dataloader()

  LightningModule = create_lightning_module(config).to('cuda:0')

  with torch.no_grad():
    total_sum = None
    total_num_images = 0
    max_val = float('-inf')
    min_val = float('inf')
    for i, batch in tqdm(enumerate(train_dataloader)):
        hf = LightningModule.get_hf_coefficients(batch.to('cuda:0'))

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


def multi_scale_test(master_config, log_path):
  logger = pl.loggers.TensorBoardLogger(log_path, name='autoregressive_samples')

  scale_info = {}
  for config_name, config in master_config.items():
    scale = config.data.image_size
    scale_info[scale] = {}

    DataModule = create_lightning_datamodule(config)
    scale_info[scale]['DataModule'] = DataModule

    LightningModule = create_lightning_module(config)
    LightningModule = LightningModule.load_from_checkpoint(config.model.checkpoint_path)
    LightningModule.configure_sde(config, sigma_max_y = LightningModule.sigma_max_y)

    scale_info[scale]['LightningModule'] = LightningModule.to('cuda:0')
    scale_info[scale]['LightningModule'].eval()
    
  def get_autoregressive_sampler(scale_info):
    def autoregressive_sampler(dc, return_intermediate_images = False, show_evolution = False):
      if return_intermediate_images:
        scales_dc = []
        scales_dc.append(dc)
      
      if show_evolution:
        scale_evolutions = {'haar':[], 'image':[]}

      for count, scale in enumerate(sorted(scale_info.keys())):
        lightning_module = scale_info[scale]['LightningModule']
        print('sigma_max_y: %.4f' % lightning_module.sigma_max_y)
        print('lightning_module.sde[0].sigma_max: ', lightning_module.sde[0].sigma_max)
        print('lightning_module.device: ', lightning_module.device)
        print('dc.device: ', dc.device)
        hf, info = lightning_module.sample(dc, show_evolution) #inpaint the high frequencies of the next resolution level

        if show_evolution:
          evolution = info['evolution']
          dc = dc.to('cpu')

          haar_grid_evolution = []
          for frame in range(evolution['x'].size(0)):
            haar_grid_evolution.append(create_supergrid(normalise_per_band(torch.cat((dc, evolution['x'][frame]), dim=1))))

          dc = dc.to('cuda')

        haar_image = torch.cat([dc,hf], dim=1)
        dc = lightning_module.haar_backward(haar_image) #inverse the haar transform to get the dc coefficients of the new scale

        if show_evolution:
          if count == len(scale_info.keys()) - 1:
            image_grid = make_grid(normalise_per_image(dc.to('cpu')), nrow=int(np.sqrt(dc.size(0))))
            haar_grid_evolution.append(image_grid)
          
          haar_grid_evolution = torch.stack(haar_grid_evolution)
          scale_evolutions['haar'].append(haar_grid_evolution)

        if return_intermediate_images:
          scales_dc.append(dc)

      #return output logic here
      if return_intermediate_images and show_evolution:
        return scales_dc, scale_evolutions
      elif return_intermediate_images and not show_evolution:
         return scales_dc, []
      elif not return_intermediate_images and show_evolution:
          return [], scale_evolutions
      else:
        return dc, []

    return autoregressive_sampler
  
  autoregressive_sampler = get_autoregressive_sampler(scale_info)

  smallest_scale = min(list(scale_info.keys()))
  smallest_scale_lightning_module = scale_info[smallest_scale]['LightningModule']
  smallest_scale_datamodule = scale_info[smallest_scale]['DataModule']
  smallest_scale_datamodule.setup()
  test_dataloader = smallest_scale_datamodule.test_dataloader()
  

  def rescale_and_concatenate(intermediate_images):
    #rescale all images to the highest detected resolution with NN interpolation and normalise them
    max_sr_factor = 2**(len(intermediate_images)-1)

    upsampled_images = []
    for i, image in enumerate(intermediate_images):
      if i == len(intermediate_images)-1:
        upsampled_images.append(normalise_per_image(image)) #normalise and append
      else:
        upsample_fn = Upsample(scale_factor=max_sr_factor/2**i, mode='nearest') #upscale to the largest resolution
        upsampled_image = upsample_fn(image)
        upsampled_images.append(normalise_per_image(upsampled_image)) #normalise and append
    
    concat_upsampled_images = torch.cat(upsampled_images, dim=-1)
    
    return concat_upsampled_images

  def create_scale_evolution_video(scale_evolutions):
    total_frames = sum([evolution.size(0) for evolution in scale_evolutions])
    
    #initialise the concatenated tensor video and then fill it
    concat_video = torch.zeros(size=tuple([total_frames,]+list(scale_evolutions[-1].shape[1:])))
    print(concat_video.size())

    previous_last_frame = 0
    new_last_frame = 0
    for evolution in scale_evolutions:
      new_last_frame += evolution.size(0)
      print(new_last_frame, previous_last_frame)
      concat_video[previous_last_frame:new_last_frame, :evolution.size(1), :evolution.size(2), :evolution.size(3)] = evolution
      previous_last_frame = new_last_frame

    return concat_video

  for i, batch in enumerate(test_dataloader):
    batch = smallest_scale_lightning_module.get_dc_coefficients(batch.to('cuda:0'))
    intermediate_images, scale_evolutions = autoregressive_sampler(batch, return_intermediate_images=True, show_evolution=False)
    concat_upsampled_images = rescale_and_concatenate(intermediate_images)
    
    concat_grid = make_grid(concat_upsampled_images, nrow=int(np.sqrt(concat_upsampled_images.size(0))))
    logger.experiment.add_image('Autoregressive_Sampling_batch_%d' % i, concat_grid)

    #concat_video = create_scale_evolution_video(scale_evolutions['haar']).unsqueeze(0)
    #logger.experiment.add_video('Autoregressive_Sampling_evolution_batch_%d' % i, concat_video, fps=50)


