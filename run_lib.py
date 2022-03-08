from models import ddpm, ncsnv2, fcn, ddpm3D, ncsnpp #needed for model registration
import pytorch_lightning as pl
#from pytorch_lightning.plugins import DDPPlugin
import numpy as np

from torchvision.utils import make_grid

from losses import get_distillation_loss_fn

from lightning_callbacks.DistillationCallback import DistillationCallback
from lightning_callbacks import callbacks, HaarMultiScaleCallback, PairedCallback #needed for callback registration
from lightning_callbacks.HaarMultiScaleCallback import normalise_per_image, permute_channels, normalise, normalise_per_band, create_supergrid
from lightning_callbacks.utils import get_callbacks

from lightning_data_modules import HaarDecomposedDataset, ImageDatasets, PairedDataset, SyntheticDataset, SRDataset, SRFLOWDataset, DUALGLOWDataset #needed for datamodule registration
from lightning_data_modules.utils import create_lightning_datamodule

from lightning_modules import MultiScaleSdeGenerativeModel, DistillationModel, BaseSdeGenerativeModel, HaarMultiScaleSdeGenerativeModel, ConditionalSdeGenerativeModel #need for lightning module registration
from lightning_modules.utils import create_lightning_module

from torchvision.transforms import RandomCrop, CenterCrop, ToTensor, Resize
from torchvision.transforms.functional import InterpolationMode

from evaluation import run_conditional_evaluation_pipeline, run_unconditional_evaluation_pipeline
import create_dataset
import compute_dataset_statistics
from torch.nn import Upsample
import torch 

from pathlib import Path
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from fast_sampling.computation_utils import fast_sampling_scheme, get_Lip_constant_profile

def compute_fast_sampling_scheme(config, save_dir):
  fast_sampling_scheme(config, save_dir)

def compute_lipschitz_constant(config):
  get_Lip_constant_profile(config)

def run_distillation(config):
  DataModule = create_lightning_datamodule(config)
  starting_iter = config.distillation.starting_iter #default would be 1

  for it in range(starting_iter, config.distillation.iterations+1):
    logger = pl.loggers.TensorBoardLogger(config.distillation.log_path, name='distillation_it_%d' % it)
    
    if it == 1:
      Dmodule = DistillationModel.BaseDistillationModel(config)
      TeacherModule = create_lightning_module(config)
      TeacherModule = TeacherModule.load_from_checkpoint(config.model.checkpoint_path)
      Dmodule.TeacherModule.load_state_dict(TeacherModule.state_dict())
      Dmodule.StudentModule.load_state_dict(Dmodule.TeacherModule.state_dict())
      Dmodule.configure_sde(config)

    elif it > 1 and it == starting_iter:
      assert config.distillation.prev_checkpoint_path is not None, 'Latest distillation teacher should be provided for initialisation from starting iteration %d' % starting_iter
      Dmodule = DistillationModel.BaseDistillationModel(config)
      Dmodule = Dmodule.load_from_checkpoint(config.distillation.prev_checkpoint_path)
      Dmodule.TeacherModule.load_state_dict(Dmodule.StudentModule.state_dict())
      Dmodule.configure_sde(config)
    
    Dmodule.iteration = it
      
    trainer = pl.Trainer(gpus=config.training.gpus,
                          num_nodes = config.training.num_nodes,
                          accelerator = config.training.accelerator,
                          accumulate_grad_batches = config.training.accumulate_grad_batches,
                          gradient_clip_val = config.distillation.optim.grad_clip,
                          max_steps=config.distillation.num_steps,
                          callbacks = DistillationCallback(config),
                          logger = logger,
                          resume_from_checkpoint = config.distillation.resume_checkpoint_path)

    trainer.fit(Dmodule, datamodule = DataModule)
    trainer.test(Dmodule, datamodule = DataModule)

    Dmodule.TeacherModule.load_state_dict(Dmodule.StudentModule.state_dict())
    Dmodule.N = Dmodule.N // 2

def train_multiscale(configs):
  base_config = configs.d1
  DataModule = create_lightning_datamodule(base_config)
  callbacks = get_callbacks(base_config)

  LightningModule = create_lightning_module(configs)

  if base_config.experiment_name is None:
      experiment_name = 'lightning_logs'
  else:
      experiment_name = base_config.experiment_name

  if base_config.base_log_path is not None:
      log_path = base_config.base_log_path
  
  logger = pl.loggers.TensorBoardLogger(log_path, name=experiment_name)

  checkpoint_path = base_config.model.checkpoint_path

  trainer = pl.Trainer(gpus=base_config.training.gpus,
                          num_nodes = base_config.training.num_nodes,
                          accelerator = base_config.training.accelerator,
                          accumulate_grad_batches = base_config.training.accumulate_grad_batches,
                          gradient_clip_val = base_config.optim.grad_clip,
                          max_steps=base_config.training.n_iters, 
                          callbacks=callbacks, 
                          logger = logger,
                          resume_from_checkpoint=checkpoint_path)

  trainer.fit(LightningModule, datamodule=DataModule)

def test_multiscale(configs):
    base_config = configs.d1
    DataModule = create_lightning_datamodule(base_config)
    DataModule.setup()

    callbacks = get_callbacks(base_config, phase='test')

    eval_log_path = base_config.base_log_path
    Path(eval_log_path).mkdir(parents=True, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=eval_log_path, name='test_metrics')

    checkpoint_path = base_config.model.checkpoint_path
    LightningModule = create_lightning_module(configs)
    trainer = pl.Trainer(gpus = base_config.training.gpus,
                          num_nodes = base_config.training.num_nodes,
                          accelerator = base_config.training.accelerator,
                          accumulate_grad_batches = base_config.training.accumulate_grad_batches,
                          gradient_clip_val = base_config.optim.grad_clip,
                          max_steps=base_config.training.n_iters, 
                          callbacks=callbacks, 
                          logger = logger,
                          resume_from_checkpoint=checkpoint_path)
    
    trainer.test(LightningModule, test_dataloaders = DataModule.test_dataloader())

    #evaluate FID scores on the generated samples
    unconditional_evaluation_pipeline(base_config)
    
def train(config, log_path, checkpoint_path):
    if config.data.create_dataset:
      create_dataset.create_dataset(config)

    DataModule = create_lightning_datamodule(config)
    callbacks = get_callbacks(config)
    LightningModule = create_lightning_module(config)

    if config.experiment_name is None:
      experiment_name = 'lightning_logs'
    else:
      experiment_name = config.experiment_name
    
    if config.base_log_path is not None:
      log_path = config.base_log_path

    logger = pl.loggers.TensorBoardLogger(log_path, name=experiment_name)

    if checkpoint_path is not None or config.model.checkpoint_path is not None:
      if config.model.checkpoint_path is not None and checkpoint_path is None:
        checkpoint_path = config.model.checkpoint_path

      trainer = pl.Trainer(gpus=config.training.gpus,
                          num_nodes = config.training.num_nodes,
                          accelerator = config.training.accelerator, #plugins = DDPPlugin(find_unused_parameters=False) if config.training.accelerator=='ddp' else None,
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
    eval_log_path = config.base_log_path
    Path(eval_log_path).mkdir(parents=True, exist_ok=True)
    logger = pl.loggers.TensorBoardLogger(save_dir=eval_log_path, name='test_metrics')

    DataModule = create_lightning_datamodule(config)
    DataModule.setup()

    callbacks = get_callbacks(config, phase='test')

    if checkpoint_path is not None or config.model.checkpoint_path is not None:
      if config.model.checkpoint_path is not None and checkpoint_path is None:
        checkpoint_path = config.model.checkpoint_path
    else:
      return 'Testing cannot be completed because no checkpoint has been provided.'

    LightningModule = create_lightning_module(config, checkpoint_path)

    trainer = pl.Trainer(gpus=config.training.gpus,
                         num_nodes = config.training.num_nodes,
                         accelerator = config.training.accelerator,
                         accumulate_grad_batches = config.training.accumulate_grad_batches,
                         gradient_clip_val = config.optim.grad_clip,
                         max_steps=config.training.n_iters, 
                         callbacks=callbacks, 
                         logger = logger)
    
    trainer.test(LightningModule, test_dataloaders = DataModule.test_dataloader())

    #evaluate FID scores on the generated samples
    unconditional_evaluation_pipeline(config)


def conditional_evaluation_pipeline(master_config):
  for config_name, config in master_config.items():
    print('Tested Configuration: %s - %s - %s' % (config.data.task, config.data.dataset, config.training.conditioning_approach))
    for snr in config.eval.snr:
      base_path = os.path.join(config.eval.base_log_dir, config.data.task, config.data.dataset, config.training.conditioning_approach, 'images')
      run_conditional_evaluation_pipeline(config.data.task, base_path, snr, device='cuda')

def unconditional_evaluation_pipeline(config):
  run_unconditional_evaluation_pipeline(config)

def multi_scale_test(master_config, log_path):
  def get_lowest_level_fn(scale_info, coord_space):
    def level_dc_coefficients_fn(batch, level=None):
      #get the dc coefficients at input level of the haar transform
      #batch is assumed to be at the highest resolution
      #if input level is None, then get the dc coefficients until the last level

      if level == None:
        level = len(scale_info.keys())
      
      for count, scale in enumerate(sorted(scale_info.keys(), reverse=True)): #start from the highest resolution
        if count == level:
          break

        lightning_module = scale_info[scale]['LightningModule']
        batch = lightning_module.get_dc_coefficients(batch)
      
      return batch
    
    def bicubic_downsampling_fn(batch):
      target_level = len(scale_info.keys())
      resize_fn = Resize(batch.size(-1)//2**target_level, interpolation=InterpolationMode.BICUBIC)
      batch = resize_fn(batch)
      return batch

    if coord_space == 'bicubic':
      return bicubic_downsampling_fn
    elif coord_space == 'haar':
      return level_dc_coefficients_fn
    else:
      return NotImplementedError('%s space is not supported for sequential downsampling.' % coord_space)

  def get_autoregressive_sampler(scale_info, coord_space='bicubic', 
                                 predictor='default', corrector='default', 
                                 p_steps='default', c_steps='default'):

    def bicubic_autoregressive_sampler(lr, return_intermediate_images = True,  show_evolution = False):
      if return_intermediate_images:
        scales_bicubic = []
        scales_bicubic.append(lr.clone().cpu())
      
      for count, scale in enumerate(sorted(scale_info.keys())):
        lightning_module = scale_info[scale]['LightningModule']
        lr, info = lightning_module.sample(lr, show_evolution, predictor, corrector, p_steps, c_steps)
        if return_intermediate_images:
          scales_bicubic.append(lr.clone().cpu())
      
      if return_intermediate_images:
        return scales_bicubic, []
      else:
        return lr, []

    def haar_autoregressive_sampler(dc, return_intermediate_images = False, show_evolution = False):
      if return_intermediate_images:
        scales_dc = []
        scales_dc.append(dc.clone().cpu())
      
      if show_evolution:
        scale_evolutions = {'haar':[], 'image':[]}

      for count, scale in enumerate(sorted(scale_info.keys())):
        lightning_module = scale_info[scale]['LightningModule']
        print('sigma_max_y: %.4f' % lightning_module.sigma_max_y)
        print('lightning_module.sde.sigma_max: ', lightning_module.sde['y'].sigma_max)
        print('lightning_module.device: ', lightning_module.device)
        print('dc.device: ', dc.device)

        #inpaint the high frequencies of the next resolution level
        hf, info = lightning_module.sample(dc, show_evolution, predictor, corrector, p_steps, c_steps) 

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
          scales_dc.append(dc.clone().cpu())

      #return output logic here
      if return_intermediate_images and show_evolution:
        return scales_dc, scale_evolutions
      elif return_intermediate_images and not show_evolution:
         return scales_dc, []
      elif not return_intermediate_images and show_evolution:
          return [], scale_evolutions
      else:
        return dc, []

    if coord_space == 'bicubic':
      return bicubic_autoregressive_sampler
    elif coord_space == 'haar':
      return haar_autoregressive_sampler
    else:
      return NotImplementedError('%s space is not supported for autoregressive sampling.' % coord_space)
  
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


  #script code for multi_scale testing starts here.
  #create the loggger
  logger = pl.loggers.TensorBoardLogger(log_path, name='autoregressive_samples') 

  #store the models, dataloaders and configure sdes (especially the conditioning sde) for all scales
  scale_info = {}
  for config_name, config in master_config.items():
    scale = config.data.image_size
    scale_info[scale] = {}

    coord_space = config.data.coordinate_space
    DataModule = create_lightning_datamodule(config)
    scale_info[scale]['DataModule'] = DataModule
    
    LightningModule = create_lightning_module(config)
    LightningModule = LightningModule.load_from_checkpoint(config.model.checkpoint_path)
    LightningModule.configure_sde(config, sigma_max_y = LightningModule.sigma_max_y)

    scale_info[scale]['LightningModule'] = LightningModule.to('cuda:0')
    scale_info[scale]['LightningModule'].eval()
  
  #instantiate the autoregressive sampling function
  autoregressive_sampler = get_autoregressive_sampler(scale_info, coord_space, p_steps=2000, corrector='conditional_none')

  #instantiate the function that computes the dc coefficients of the input batch at the required depth/level.
  #lowest_level_fn = get_lowest_level_fn(scale_info, coord_space)

  #get test dataloader of the highest scale
  max_scale = max(list(scale_info.keys()))
  max_scale_datamodule = scale_info[max_scale]['DataModule']
  max_scale_datamodule.setup()
  max_test_dataloader = max_scale_datamodule.test_dataloader()
  max_test_batch = max_scale_datamodule.test_batch

  #get test dataloader of the minimum scale
  min_scale = min(list(scale_info.keys()))
  min_scale_datamodule = scale_info[min_scale]['DataModule']
  min_scale_datamodule.setup()
  min_scale_datamodule.test_batch = max_test_batch
  min_test_dataloader = min_scale_datamodule.test_dataloader()
  
  #iterate over the test dataloader of the highest scale
  for i, (batch_lr, batch_hr) in enumerate(zip(min_test_dataloader, max_test_dataloader)):

    '''
    if coord_space == 'haar':
      lr2x, hr = batch
      hr_batch = hr.clone().cpu()
      batch = hr
    elif coord_space == 'bicubic':
      lr2x, hr = batch
      hr_batch = hr.clone().cpu()
      batch = hr
    
    batch = lowest_level_fn(batch.to('cuda:0')) #compute the DC/Bicubic coefficients at maximum depth (smallest resolution)
    '''

    lr = batch_lr[0].to('cuda:0')

    if coord_space == 'haar':
      hr = scale_info[max_scale]['LightningModule'].haar_backward(torch.cat(batch_hr, dim=1).to('cuda:0')).cpu()
    else:
      hr = batch_hr[1].cpu()

    intermediate_images, scale_evolutions = autoregressive_sampler(lr, return_intermediate_images=True, show_evolution=False)
    concat_upsampled_images = rescale_and_concatenate(intermediate_images)

    vis_concat = torch.cat((concat_upsampled_images, normalise_per_image(hr)), dim=-1) #concatenated intermediate images and the GT hr batch
    
    concat_grid = make_grid(vis_concat, nrow=1, normalize=False)
    logger.experiment.add_image('Autoregressive_Sampling_batch_%d' % i, concat_grid)

    #concat_video = create_scale_evolution_video(scale_evolutions['haar']).unsqueeze(0)
    #logger.experiment.add_video('Autoregressive_Sampling_evolution_batch_%d' % i, concat_video, fps=50)


def compute_data_stats(config):
  compute_dataset_statistics.compute_dataset_statistics(config)