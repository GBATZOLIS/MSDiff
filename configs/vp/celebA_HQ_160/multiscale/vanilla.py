import ml_collections
import torch
import math
import numpy as np

def get_config():
  config = ml_collections.ConfigDict()

  #logging
  config.base_log_path = '/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/projects/fast_reverse_diffusion/multiscale/debug_vanilla' #'/home/gb511/projects/fast_sampling' 
  config.experiment_name = 'loglinear_128'

  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.lightning_module = 'base'
  training.batch_size = 64
  training.num_nodes = 1
  training.gpus = 1
  training.accelerator = None if training.gpus == 1 else 'ddp'
  training.accumulate_grad_batches = 1
  training.workers = 4*training.gpus
  training.num_epochs = 10000
  training.n_iters = 1000000 
  training.visualization_callback = 'base'
  training.show_evolution = False
  
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = True
  training.continuous = True
  training.reduce_mean = True 
  training.sde = 'snrvpsde'
  
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
  sampling.lipschitz_profile = '/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/projects/fast_reverse_diffusion/celebA-HQ-160/vp/vp_celebA_smld_weighting/Lip_constant/info.pkl' #'/home/gb511/projects/fast_sampling/vp_celebA_smld_weighting/Lip_constant/info.pkl'

  # evaluation (this file is not modified at all - subject to change)
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.workers = 4*training.gpus
  evaluate.batch_size = training.batch_size
  evaluate.callback = 'base'
  evaluate.predictor = 'euler_maruyama'
  evaluate.corrector = 'none'
  evaluate.p_steps = [100] #[100, 200, 400, 800] #np.arange(100, 1100, step=100)
  evaluate.c_steps = 1
  evaluate.probability_flow = True
  evaluate.denoise = True
  evaluate.adaptive = [False, True] 
  evaluate.adaptive_method = 'lipschitz' #options: [kl, lipschitz]
  evaluate.alpha = [1.] #used for lipschitz-adaptive method
  evaluate.starting_T = [1.]
  evaluate.gamma = [1.] #0->uniform, 1->KL-adaptive #used for the KL-adaptive method
  evaluate.num_samples = 10000

  # data
  config.data = data = ml_collections.ConfigDict()
  data.base_dir = '/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/datasets' #'datasets'
  data.dataset = 'celebA-HQ-160'
  data.use_data_mean = False
  data.datamodule = 'unpaired_PKLDataset'
  data.create_dataset = False
  data.split = [0.8, 0.1, 0.1]
  data.image_size = 128
  data.effective_image_size = data.image_size
  data.shape = [3, data.image_size, data.image_size]
  data.centered = False
  data.random_flip = False
  data.crop = False
  data.uniform_dequantization = False
  data.num_channels = data.shape[0] #the number of channels the model sees as input.

  # model
  config.model = model = ml_collections.ConfigDict()
  model.checkpoint_path = '/home/gb511/rds/rds-t2-cs138-LlrDsbHU5UM/gb511/projects/fast_reverse_diffusion/multiscale/debug_vanilla/loglinear_128/version_0/checkpoints/epoch=43-step=111935.ckpt'
  model.num_scales = 1000
  model.sigma_max = np.sqrt(np.prod(data.shape))
  model.sigma_min = 0.01
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'

   # model architecture
  model.name = 'ddpm'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult =  (1, 1, 2, 2, 4, 4) #(1, 1, 2, 2, 4)
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
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.beta2 = 0.99
  optim.eps = 1e-8
  optim.warmup = 5000 #set it to 0 if you do not want to use warm up.
  optim.grad_clip = 1 #set it to 0 if you do not want to use gradient clipping using the norm algorithm. Gradient clipping defaults to the norm algorithm.
  config.seed = 42

  # distillation
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

  return config