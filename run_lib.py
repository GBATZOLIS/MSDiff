from models import ddpm, ncsnv2, fcn #needed for model registration
import pytorch_lightning as pl

from torchvision.utils import make_grid

from lightning_callbacks import callbacks, HaarMultiScaleCallback, PairedCallback #needed for callback registration
from lightning_callbacks.HaarMultiScaleCallback import normalise_per_image, permute_channels
from lightning_callbacks.utils import get_callbacks

from lightning_data_modules import HaarDecomposedDataset, ImageDatasets, PairedDataset, SyntheticDataset #needed for datamodule registration
from lightning_data_modules.utils import create_lightning_datamodule

from lightning_modules import BaseSdeGenerativeModel, HaarMultiScaleSdeGenerativeModel, ConditionalSdeGenerativeModel #need for lightning module registration
from lightning_modules.utils import create_lightning_module

import create_dataset
from torch.nn import Upsample
import torch 

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
                          accumulate_grad_batches = config.training.accumulate_grad_batches,
                          gradient_clip_val = config.optim.grad_clip,
                          max_steps=config.training.n_iters, 
                          callbacks=callbacks, 
                          logger = logger,
                          resume_from_checkpoint=checkpoint_path)
    else:  
      trainer = pl.Trainer(gpus=config.training.gpus,
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

  callbacks = get_callbacks(config, phase='test')
  LightningModule = create_lightning_module(config)
  logger = pl.loggers.TensorBoardLogger(log_path, name='test_lightning_logs')

  if checkpoint_path is not None:
      trainer = pl.Trainer(gpus=config.training.gpus,
                          accumulate_grad_batches = config.training.accumulate_grad_batches,
                          gradient_clip_val = config.optim.grad_clip,
                          max_steps=config.training.n_iters, 
                          callbacks=callbacks, 
                          logger = logger,
                          limit_val_batches=1,
                          resume_from_checkpoint=checkpoint_path)
  
  trainer.fit(LightningModule, datamodule=DataModule)

  print(LightningModule.sigma_max_y)
  #trainer.test(LightningModule, DataModule.test_dataloader())

def multi_scale_test(master_config, log_path):
  logger = pl.loggers.TensorBoardLogger(log_path, name='autoregressive_samples')

  scale_info = {}
  for config_name, config in master_config.items():
    scale = config.data.image_size
    scale_info[scale] = {}

    DataModule = create_lightning_datamodule(config)
    scale_info[scale]['DataModule'] = DataModule

    callbacks = get_callbacks(config, phase='test')

    LightningModule = create_lightning_module(config)

    assert config.model.checkpoint_path is not None, 'Checkpoint path is not provided'
    trainer = pl.Trainer(gpus=config.training.gpus,
                        accumulate_grad_batches = config.training.accumulate_grad_batches,
                        gradient_clip_val = config.optim.grad_clip,
                        max_steps=config.training.n_iters, 
                        callbacks=callbacks, 
                        logger = logger,
                        limit_val_batches=1,
                        resume_from_checkpoint=config.model.checkpoint_path)
    
    trainer.fit(LightningModule, datamodule=DataModule) #proper resuming of training state
    scale_info[scale]['LightningModule'] = LightningModule.to('cuda:0')
    scale_info[scale]['LightningModule'].eval()
    
  def get_autoregressive_sampler(scale_info):
    def autoregressive_sampler(dc, return_intermediate_images = False):
      if return_intermediate_images:
        scales_dc = []
        scales_dc.append(dc)

      for scale in sorted(scale_info.keys()):
        lightning_module = scale_info[scale]['LightningModule']
        print('sigma_max_y: %.4f' % lightning_module.sigma_max_y)
        print(lightning_module.sde[0].sigma_max)
        hf, _ = lightning_module.sample(dc) #inpaint the high frequencies of the next resolution level
        haar_image = torch.cat([dc,hf], dim=1)
        dc = lightning_module.haar_backward(haar_image) #inverse the haar transform to get the dc coefficients of the new scale

        if return_intermediate_images:
          scales_dc.append(dc)

      if return_intermediate_images:
        return scales_dc
      else:
        return dc

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

  for i, batch in enumerate(test_dataloader):
    batch = smallest_scale_lightning_module.get_dc_coefficients(batch.to('cuda:0'))
    intermediate_images = autoregressive_sampler(batch, return_intermediate_images=True)
    concat_upsampled_images = rescale_and_concatenate(intermediate_images)
    concat_grid = make_grid(concat_upsampled_images, nrow=int(np.sqrt(concat_upsampled_images.size(0))))
    print(concat_grid.size())
    logger.experiment.add_image('Autoregressive_Sampling_batch_%d' % i, concat_grid)

    