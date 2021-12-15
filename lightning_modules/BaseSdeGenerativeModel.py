import losses
from losses import get_sde_loss_fn, get_smld_loss_fn, get_ddpm_loss_fn, get_general_sde_loss_fn
import pytorch_lightning as pl
import sde_lib
from sampling.unconditional import get_sampling_fn
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from . import utils
import torch.optim as optim
import os
import torch
from fast_sampling.computation_utils import get_adaptive_discretisation_fn
import pickle 

@utils.register_lightning_module(name='base')
class BaseSdeGenerativeModel(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the score model
        self.config = config
        self.score_model = mutils.create_model(config)

    def configure_sde(self, config):
        # Setup SDEs
        if config.training.sde.lower() == 'vpsde':
            self.sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'subvpsde':
            self.sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'vesde':
            if config.data.use_data_mean:
                data_mean_path = os.path.join(config.data.base_dir, 'datasets_mean', '%s_%d' % (config.data.dataset, config.data.image_size), 'mean.pt')
                data_mean = torch.load(data_mean_path)
            else:
                data_mean = None
            self.sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales, data_mean=data_mean)
            self.sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    def configure_loss_fn(self, config, train):
        if config.training.continuous:
            loss_fn = get_general_sde_loss_fn(self.sde, train, reduce_mean=config.training.reduce_mean,
                                    continuous=True, likelihood_weighting=config.training.likelihood_weighting)
        else:
            if isinstance(self.sde, VESDE):
                loss_fn = get_smld_loss_fn(self.sde, train, reduce_mean=config.training.reduce_mean, likelihood_weighting=config.training.likelihood_weighting)
            elif isinstance(self.sde, VPSDE):
                assert not config.training.likelihood_weighting, "Likelihood weighting is not supported for original DDPM training."
                loss_fn = get_ddpm_loss_fn(self.sde, train, reduce_mean=config.training.reduce_mean)
            else:
                raise ValueError(f"Discrete training for {self.sde.__class__.__name__} is not recommended.")
        
        return loss_fn

    def training_step(self, batch, batch_idx):
        loss = self.train_loss_fn(self.score_model, batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.eval_loss_fn(self.score_model, batch)
        self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        return 
    
    def sample(self, show_evolution=False, num_samples=None, predictor='default', 
                    corrector='default', p_steps='default', c_steps='default', 
                    snr='default', denoise='default', adaptive='default'):
        
        if num_samples is None:
            num_samples = self.config.eval.batch_size
        
        #Code for managing adaptive sampling
        if adaptive == 'default':
            adaptive = self.config.sampling.adaptive
            assert adaptive in [True, False], 'adaptive flag should be either True or False'

        if adaptive:
            try:
                assert hasattr(self.config.sampling, 'kl_profile'), 'config.sampling.kl_profile must be provided if adaptive is set to True.'
                if self.config.sampling.kl_profile == None:
                    adaptive = False
            except AssertionError:
                adaptive = False #set adaptive to False since we cannot use it given that we are not provided with the KL profile

            if adaptive:
                print('KL adaptive discretisation is used.')
                try:
                    adaptive_discretisation_fn = self.adaptive_dicrete_fn
                except AttributeError:
                    #load the KL profile
                    with open(self.config.sampling.kl_profile, 'rb') as f:
                        info = pickle.load(f)
                    
                    adaptive_discretisation_fn = get_adaptive_discretisation_fn(info['t'], info['KL'])
                    self.adaptive_dicrete_fn = adaptive_discretisation_fn

            else:
                print('uniform-discretisation is used.')
                adaptive_discretisation_fn=None 

        sampling_shape = [num_samples] + self.config.data.shape
        sampling_fn = get_sampling_fn(config=self.config, 
                                      sde=self.sde, 
                                      shape=sampling_shape, 
                                      eps=self.sampling_eps,
                                      predictor=predictor, 
                                      corrector=corrector, 
                                      p_steps=p_steps, 
                                      c_steps=c_steps, 
                                      snr=snr, 
                                      denoise=denoise, 
                                      adaptive_disc_fn=adaptive_discretisation_fn)

        return sampling_fn(self.score_model, show_evolution=show_evolution)

    def configure_optimizers(self):
        class scheduler_lambda_function:
            def __init__(self, warm_up):
                self.use_warm_up = True if warm_up > 0 else False
                self.warm_up = warm_up

            def __call__(self, s):
                if self.use_warm_up:
                    if s < self.warm_up:
                        return s / self.warm_up
                    else:
                        return 1
                else:
                    return 1
        

        optimizer = losses.get_optimizer(self.config, self.score_model.parameters())
        
        scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(optimizer,scheduler_lambda_function(self.config.optim.warmup)),
                    'interval': 'step'}  # called after each training step
                    
        return [optimizer], [scheduler]

    
    

