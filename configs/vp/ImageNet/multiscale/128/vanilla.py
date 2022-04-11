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
    config.base_log_path = '/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/projects/fast_reverse_diffusion/multiscale/ImageNet/%d' % image_size 
  elif server == 'abg':
    config.base_log_path = '/home/gb511/projects/fast_sampling/ImageNet/%d' % image_size

  config.experiment_name = 'vanilla'

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.lightning_module = 'base'
  training.num_nodes = 1
  training.gpus = 2
  training.batch_size = 128 // (training.num_nodes*training.gpus)
  training.accelerator = None if training.gpus == 1 else 'ddp'
  training.accumulate_grad_batches = 1
  training.workers = 4*training.gpus
  training.num_epochs = 10000
  training.n_iters = 1000000
  training.visualization_callback = 'base'
  training.show_evolution = False

  #Model checkpointing
  training.checkpointing_strategy = 'mixed' #options: [mixed, last]
  training.latest_save_every_n_train_steps = 100 #replace
  training.save_every_n_train_steps = 500 #save all
  

  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = True 
  training.sde = 'vpsde' #vpsde means beta-linear vp sde
  
  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'none'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.075 #0.15 in VE sde (you typically need to play with this term - more details in the main paper)
  
  #new additions for adaptive sampling
  sampling.adaptive = False
  #provide the directory where the information needed for calculating the adaptive steps is saved.
  sampling.kl_profile = None
  sampling.lipschitz_profile = None

  # evaluation (this file is not modified at all - subject to change)
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.workers = 4*training.gpus
  evaluate.batch_size = training.batch_size
  evaluate.callback = 'base'

  evaluate.num_samples = 10000
  evaluate.probability_flow = True
  evaluate.predictor = ['euler_trapezoidal_s_2_a_0']
  evaluate.corrector = 'none'
  evaluate.p_steps = [128] 
  evaluate.c_steps = 1
  evaluate.denoise = True

  evaluate.adaptive = False
  evaluate.adaptive_method = 'lipschitz' #options: [kl, lipschitz]
  evaluate.alpha = [1.] #used for lipschitz-adaptive method
  evaluate.starting_T = [1.]
  evaluate.gamma = [1.] #0->uniform, 1->KL-adaptive #used for the KL-adaptive method
  

  # data
  config.data = data = ml_collections.ConfigDict()

  if server == 'hpc':
    data.base_dir = '/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/datasets' 
  elif server == 'abg':
    data.base_dir =  '/home/gb511/datasets/ILSVRC/Data'
 
  data.datamodule = 'ImageNet'
  data.image_size = image_size
  data.dataset = 'ImageNet_%d' % image_size
  data.use_data_mean = False
  data.create_dataset = False
  data.split = [0.8, 0.1, 0.1]
  data.effective_image_size = data.image_size
  data.shape = [3, data.image_size, data.image_size]
  data.dims = len(data.shape[1:])
  data.centered = False
  data.random_flip = False
  data.crop = False
  data.uniform_dequantization = False
  data.num_channels = data.shape[0] #the number of channels the model sees as input.

  # model
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = None
  model.num_scales = 1000
  model.sigma_max = np.sqrt(np.prod(data.shape))
  model.sigma_min = 0.01
  model.beta_min = 0.1
  model.beta_max = 20.
  
  # model architecture
  model.name = 'guided_diffusion_UNET'
  model.model_channels = 256
  model.input_channels = data.num_channels
  model.output_channels = data.num_channels
  model.num_res_blocks = 2
  model.attention_resolutions = (32, 16, 8)
  model.dropout = 0.
  model.channel_mult =  (1, 1, 2, 3, 4)
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

  # distillation
  '''
  config.distillation = distillation = ml_collections.ConfigDict()
  distillation.log_path = '/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/projects/fast_reverse_diffusion/celebA-HQ-160/vp/vp_celebA_smld_weighting/distillation'
  distillation.starting_iter = 2
  distillation.iterations = 9
  distillation.N = 512 // 2**(distillation.starting_iter-1)  #initial target for the student sampling steps -> will be halved at the end of every iteration
  distillation.num_steps = 50000

  #resume from the checkpoint of the previous iteration. Training from the start for the current starting iteration.
  distillation.prev_checkpoint_path = '/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/projects/fast_reverse_diffusion/celebA-HQ-160/vp/vp_celebA_smld_weighting/distillation/distillation_it_1/version_0/checkpoints/epoch=39-step=49999.ckpt' 
  
  #Training from the last checkpoint of the current starting iteration.
  distillation.resume_checkpoint_path = None

  distillation.optim = ml_collections.ConfigDict()
  distillation.optim.weight_decay = 0
  distillation.optim.optimizer = 'Adam'
  distillation.optim.lr = 2e-5
  distillation.optim.beta1 = 0.9
  distillation.optim.eps = 1e-8
  distillation.optim.warmup = 0 #set it to 0 if you do not want to use warm up.
  distillation.optim.grad_clip = 1 #set it to 0 if you do not want to use gradient clipping using the norm algorithm. Gradient clipping defaults to the norm algorithm.
  '''

  return config