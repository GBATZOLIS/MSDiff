import losses
from losses import get_sde_loss_fn, get_smld_loss_fn, get_ddpm_loss_fn
import pytorch_lightning as pl
import sde_lib
import sampling.sampling as sampling
from models.ema import ExponentialMovingAverage
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from . import utils

@utils.register_lightning_module(name='base')
class BaseSdeGenerativeModel(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the score model
        self.config = config
        self.score_model = mutils.create_model(config)
        self.ema = ExponentialMovingAverage(self.score_model.parameters(), decay=config.model.ema_rate)

        # Placeholder to store samples
        self.samples = None

        #Sampling settings
        self.data_shape = config.data.shape
        self.default_sampling_shape = [config.training.batch_size] +  self.data_shape
    
    def configure_sde(self, config):
        # Setup SDEs
        if config.training.sde.lower() == 'vpsde':
            self.sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'subvpsde':
            self.sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            self.sampling_eps = 1e-3
        elif config.training.sde.lower() == 'vesde':
            self.sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
            self.sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    def configure_loss_fn(self, config, train):
        if config.training.continuous:
            loss_fn = get_sde_loss_fn(self.sde, train, reduce_mean=config.training.reduce_mean,
                                    continuous=True, likelihood_weighting=config.training.likelihood_weighting)
        else:
            assert not config.training.likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(self.sde, VESDE):
                loss_fn = get_smld_loss_fn(self.sde, train, reduce_mean=config.training.reduce_mean)
            elif isinstance(self.sde, VPSDE):
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
    
    def sample(self, show_evolution=False, num_samples=None):
        # Construct the sampling function
        if num_samples is None:
            sampling_shape = self.default_sampling_shape
        else:
            sampling_shape = [num_samples] +  self.config.data_shape
        self.sampling_fn = sampling.get_sampling_fn(self.config, self.sde, sampling_shape, self.sampling_eps)

        return self.sampling_fn(self.score_model, show_evolution=show_evolution)

    def configure_optimizers(self):
        optimizer = losses.get_optimizer(self.config, self.score_model.parameters())
        return optimizer

    

