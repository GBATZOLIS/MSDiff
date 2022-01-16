import losses
from losses import get_distillation_loss_fn
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
from lightning_modules.utils import create_lightning_module


@utils.register_lightning_module(name='base_distillation')
class BaseDistillationModel(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.N = config.distillation.N #initial target for the student sampling steps -> will be changed in every iteration

        self.TeacherModule = create_lightning_module(config)
        self.TeacherModule.configure_sde(config)
        self.TeacherModule.freeze()
        
        self.StudentModule = create_lightning_module(config)
        self.StudentModule.configure_sde(config)
    
    def training_step(self, batch, batch_idx):
        distillation_loss_fn = get_distillation_loss_fn(N=self.N, sde=self.sde, 
                                                        train=True, continuous=self.config.training.continuous, 
                                                        eps=self.sampling_eps)
        
        loss = distillation_loss_fn(self.StudentModule.score_model, self.TeacherModule.score_model, batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        distillation_loss_fn = get_distillation_loss_fn(N=self.N, sde=self.sde, 
                                                        train=False, continuous=self.config.training.continuous, 
                                                        eps=self.sampling_eps)
        
        loss = distillation_loss_fn(self.StudentModule.score_model, self.TeacherModule.score_model, batch)
        self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss 
    
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
        

        optimizer = losses.get_optimizer(self.config.distillation, self.StudentModule.parameters())
        
        scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(optimizer, scheduler_lambda_function(self.config.distillation.optim.warmup)),
                    'interval': 'step'}  # called after each training step
                    
        return [optimizer], [scheduler]
    
    def sample(self, num_samples):
        def get_ddim_step_fn(sde, denoising_fn):
            def ddim_step_fn(z_t, t, dt):
                #t is a vector (batch_size,)
                #dt: (batch_size,)
                #z_t: (batch_size, **data_dims)
                t_dash = t + dt
                a_t, sigma_t = sde.perturbation_coefficients(t)
                a_t_dash, sigma_t_dash = sde.perturbation_coefficients(t_dash)

                sigma_ratio = sigma_t_dash / sigma_t
                z_t_factor = sigma_ratio[(...,) + (None,) * len(z_t.shape[1:])]
                
                denoising_factor = a_t_dash - sigma_ratio*a_t
                denoising_factor = denoising_factor[(...,) + (None,) * len(z_t.shape[1:])]

                z_step = z_t_factor * z_t + denoising_factor * denoising_fn(z_t, t)
                return z_step
            
            return ddim_step_fn

        shape = [num_samples] + self.config.data.shape
        model = self.StudentModule.score_model
        denoising_fn = mutils.get_denoising_fn(self.sde, model, train=False, continuous=True)
        ddim_step = get_ddim_step_fn(self.sde, denoising_fn)

        with torch.no_grad():
            x = self.sde.prior_sampling(shape).to(model.device).type(torch.float32)
            timesteps = torch.linspace(self.sde.T, self.sampling_eps, self.N+1, device=model.device)

            for i in range(self.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                dt = timesteps[i+1] - timesteps[i]
                x = ddim_step(x, vec_t, dt)
        
        return x