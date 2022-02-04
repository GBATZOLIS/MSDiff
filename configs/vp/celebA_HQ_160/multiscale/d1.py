import ml_collections
import torch
import math
import numpy as np

def get_config():
  config = ml_collections.ConfigDict()

  #logging
  config.base_log_path = '/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/projects/fast_reverse_diffusion/multiscale' #'/home/gb511/projects/fast_sampling' 
  config.experiment_name = 'multiscale'

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.lightning_module = 'multiscale_base'
  training.batch_size = 64
  training.num_nodes = 1
  training.gpus = 1
  training.accelerator = None if training.gpus == 1 else 'ddp'
  training.accumulate_grad_batches = 1
  training.workers = 4*training.gpus
  training.num_epochs = 10000
  training.n_iters = 1000000 
  training.visualization_callback = 'multiscale_base'
  training.show_evolution = False
  
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = True
  training.continuous = True
  training.reduce_mean = True 
  training.sde = 'vpsde'
  
  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'ddim'
  sampling.corrector = 'none'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.075 #0.15 in VE sde (you typically need to play with this term - more details in the main paper)
  sampling.adaptive = False

  # evaluation (this file is not modified at all - subject to change)
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.workers = 4*training.gpus
  evaluate.batch_size = training.batch_size
  evaluate.enable_sampling = True
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'
  evaluate.adaptive = False

  # data
  config.data = data = ml_collections.ConfigDict()
  data.base_dir = '/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/datasets' #'datasets'
  data.dataset = 'celebA-HQ-160'
  data.use_data_mean = False
  data.datamodule = 'unpaired_PKLDataset'
  data.create_dataset = False
  data.split = [0.8, 0.1, 0.1]
  data.centered = False
  data.random_flip = False
  data.crop = False
  data.uniform_dequantization = False

  #multiscale settings
  data.scale_depth = 1 #k
  data.scale_type = 'd'
  data.scale_name = data.scale_type + str(data.scale_depth)

  data.max_haar_depth = 3
  data.num_scales = data.max_haar_depth + 1

  #shapes used for the score model construction
  data.target_image_size = 128
  data.image_size = data.target_image_size//2**(data.scale_depth-1)
  data.effective_image_size = data.image_size
  data.shape = [3, data.image_size, data.image_size]
  data.num_channels = data.shape[0] #the number of channels the model sees as input.

  #shapes for sampling the scale coefficients
  if data.scale_name[0]=='d':
    data.scale_shape = [9, data.image_size//2, data.image_size//2]
  elif data.scale_name[0]=='a':
    data.scale_shape = data.shape

  # model
  config.model = model = ml_collections.ConfigDict()

  #multiscale settings
  model.max_haar_depth = data.max_haar_depth - (data.scale_depth-1)
  model.beta_min = 0.1
  model.T_k = data.scale_depth/data.num_scales
  target = np.exp(-1/4*(20-0.1)-1/2*0.1)
  model.beta_max = (1-2/model.T_k)*model.beta_min -4/model.T_k**2 * np.log(target/2**(data.scale_depth-1))
  
  model.checkpoint_path = None #'/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/projects/fast_reverse_diffusion/multiscale/multiscale/version_3/checkpoints/epoch=117-step=300191.ckpt'
  model.num_scales = 1000
  model.sigma_max = np.sqrt(np.prod(data.shape))
  model.sigma_min = 0.01
  model.dropout = 0.
  model.embedding_type = 'fourier'

   # model architecture
  model.name = 'ddpm_multi_speed_haar'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 1, 2, 2, 2)
  model.num_res_blocks = 2
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.input_channels = data.num_channels
  model.output_channels = data.num_channels

  # optimization
  config.optim = optim = ml_collections.ConfigDict()

  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-5
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000 #set it to 0 if you do not want to use warm up.
  optim.grad_clip = 1 #set it to 0 if you do not want to use gradient clipping using the norm algorithm. Gradient clipping defaults to the norm algorithm.
  config.seed = 42

  return config