from . import BaseSdeGenerativeModel
from losses import get_sde_loss_fn, get_smld_loss_fn, get_ddpm_loss_fn, get_inverse_problem_smld_loss_fn, get_inverse_problem_ddpm_loss_fn
from sde_lib import VESDE, VPSDE, cVESDE
from sampling.conditional import get_conditional_sampling_fn
import sde_lib
from . import utils
import torch

@utils.register_lightning_module(name='conditional')
class ConditionalSdeGenerativeModel(BaseSdeGenerativeModel.BaseSdeGenerativeModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)

    def configure_sde(self, config):
        if config.training.sde.lower() == 'vpsde':
            self.sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'subvpsde':
            self.sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'vesde':
            sde_y = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max_y, N=config.model.num_scales)
            sde_x = sde_lib.cVESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max_x, N=config.model.num_scales)
            self.sde = [sde_y, sde_x]
            self.sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
    def configure_loss_fn(self, config, train):
        if config.training.continuous:
            loss_fn = get_sde_loss_fn(self.sde, train, reduce_mean=config.training.reduce_mean,
                                    continuous=True, likelihood_weighting=config.training.likelihood_weighting)
        else:
            #assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(self.sde, list):
                #this part of the code needs to be improved. We should check all elements of the sde list. We might have a mixture.
                if isinstance(self.sde[0], VESDE) and isinstance(self.sde[1], cVESDE) and len(self.sde)==2:
                    loss_fn = get_inverse_problem_smld_loss_fn(self.sde, train, reduce_mean=config.training.reduce_mean, \
                        likelihood_weighting=config.training.likelihood_weighting)
                elif isinstance(self.sde[0], VPSDE) and isinstance(self.sde[1], VPSDE) and len(self.sde)==2:
                    loss_fn = get_inverse_problem_ddpm_loss_fn(self.sde, train, reduce_mean=config.training.reduce_mean)
                else:
                    raise NotImplementedError('This combination of sdes is not supported for discrete training yet.')
                
            elif isinstance(self.sde, VESDE):
                loss_fn = get_smld_loss_fn(self.sde, train, reduce_mean=config.training.reduce_mean)
            elif isinstance(self.sde, VPSDE):
                loss_fn = get_ddpm_loss_fn(self.sde, train, reduce_mean=config.training.reduce_mean)
            else:
                raise ValueError(f"Discrete training for {self.sde.__class__.__name__} is not recommended.")
        
        return loss_fn
    
    def configure_default_sampling_shape(self, config):
        self.data_shape = config.data.shape_x
        self.default_sampling_shape = [config.training.batch_size] +  self.data_shape

    def sample(self, y, show_evolution=False):
        sampling_shape = [y.size(0)]+self.data_shape
        conditional_sampling_fn = get_conditional_sampling_fn(self.config, self.sde, sampling_shape, self.sampling_eps)
        return conditional_sampling_fn(self.score_model, y, show_evolution)

@utils.register_lightning_module(name='conditional_decreasing_variance')
class DecreasingVarianceConditionalSdeGenerativeModel(ConditionalSdeGenerativeModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.register_buffer('sigma_max_y', torch.tensor(config.model.sigma_max_x))
    
    def configure_sde(self, config, sigma_max_y = None):
        if config.training.sde.lower() == 'vpsde':
            self.sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'subvpsde':
            self.sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'vesde':
            if sigma_max_y is None:
                sigma_max_y = config.model.sigma_max_x 

            sde_y = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=sigma_max_y, N=config.model.num_scales)
            sde_x = sde_lib.cVESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max_x, N=config.model.num_scales)
            self.sde = [sde_y, sde_x]
            self.sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")
    
    def test_step(self, batch, batch_idx):
        print('Test batch %d' % batch_idx)
