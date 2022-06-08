import ml_collections
import torch
import math
import numpy as np

def get_config():
  config = ml_collections.ConfigDict()
  image_size = 128
  server = 'hpc' #Options:['abg', 'hpc']

  #logging
  if server == 'hpc':
    config.base_log_path = '/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/projects/fast_reverse_diffusion/multiscale/celebA-HQ/OpenAI_architecture/%d' % image_size 
  elif server == 'abg':
    config.base_log_path = '/home/gb511/projects/fast_sampling/celebA-HQ/%d' % image_size

  config.experiment_name = 'multiscale'

  # training
  config.training = training = ml_collections.ConfigDict()
  training.multiscale = True
  training.lightning_module = 'multiscale_base'
  training.num_nodes = 1
  training.gpus = 1
  training.batch_size = 64 // (training.num_nodes*training.gpus)
  training.accelerator = None if training.gpus == 1 else 'ddp'
  training.accumulate_grad_batches = 1
  training.workers = 4*training.gpus
  training.num_epochs = 10000
  training.n_iters = 2796000
  training.visualization_callback = 'multiscale_base'
  training.show_evolution = False
  training.use_ema = True

  #Model checkpointing
  training.checkpointing_strategy = 'mixed' #options: [mixed, last]
  training.latest_save_every_n_train_steps = 5000 #replace
  training.save_every_n_train_steps = 250000 #save all
  
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = True
  training.continuous = True
  training.reduce_mean = True 
  training.sde = 'vpsde'
  
  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
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
  evaluate.callback = training.visualization_callback

  evaluate.checkpoint_iterations = [1999999, 2749999] #[249999, 499999, 749999, 999999, 1249999, 1499999, 1749999, 1999999]
  evaluate.checkpoint_iteration = None
  evaluate.base_checkpoint_path =  '/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/projects/fast_reverse_diffusion/multiscale/ImageNet/128/multiscale_ema/checkpoint_collection'

  evaluate.num_samples = 20000
  evaluate.probability_flow = False
  evaluate.predictor = ['ddim']
  evaluate.corrector = 'none'
  evaluate.p_steps = [128]
  evaluate.c_steps = 1
  evaluate.denoise = True

  evaluate.adaptive = False

  # data
  config.data = data = ml_collections.ConfigDict()

  if server == 'hpc':
    data.base_dir = '/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/datasets' 
  elif server == 'abg':
    data.base_dir =  '/home/gb511/datasets/ILSVRC/Data'

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
  data.dims = len(data.shape[1:])
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
  
  model.checkpoint_path = None
  model.num_scales = 1000
  model.sigma_max = np.sqrt(np.prod(data.shape))
  model.sigma_min = 0.01
  model.dropout = 0.1
  #model.embedding_type = 'fourier'

   # model architecture
  model.name = 'guided_diffusion_UNET_multi_speed_haar'
  model.model_channels = 128
  model.input_channels = data.num_channels
  model.output_channels = data.num_channels
  model.num_res_blocks = 2
  model.attention_resolutions = (32, 16, 8)
  model.dropout = 0.
  model.channel_mult =  (.5, 1, 1, 1.5, 2)
  model.conv_resample = True
  model.num_classes = None
  model.num_heads = 4
  model.num_head_channels = 64
  model.num_heads_upsample = -1
  model.use_scale_shift_norm = True
  model.use_new_attention_order = False
  model.resblock_updown = True

  model.scale_by_sigma = False
  model.ema_rate = 0.9999

  # optimization
  config.optim = optim = ml_collections.ConfigDict()

  optim.weight_decay = 0.
  optim.optimizer = 'Adam'
  optim.lr = 1e-4
  optim.beta1 = 0.9
  optim.beta2 = 0.999
  optim.eps = 1e-8
  optim.warmup = 0 #set it to 0 if you do not want to use warm up.
  optim.grad_clip = 0 #set it to 0 if you do not want to use gradient clipping using the norm algorithm. Gradient clipping defaults to the norm algorithm.
  config.seed = 42

  return config