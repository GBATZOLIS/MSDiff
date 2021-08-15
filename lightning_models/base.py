import losses
from losses import get_sde_loss_fn, get_smld_loss_fn, get_ddpm_loss_fn
import pytorch_lightning as pl
import sde_lib
import sampling
from models.ema import ExponentialMovingAverage
from models import utils as mutils
from sde_lib import VESDE, VPSDE

class BaseSdeGenerativeModel(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # Initialize model
        self.config = config
        self.score_model = mutils.create_model(config)
        self.ema = ExponentialMovingAverage(self.score_model.parameters(), decay=config.model.ema_rate)

        # Setup SDEs
        self.load_sde(self.config)
        
        # Set up loss functions
        # Build one-step training and evaluation functions
        self.continuous = config.training.continuous
        self.reduce_mean = config.training.reduce_mean
        self.likelihood_weighting = config.training.likelihood_weighting

        # Construct training losses
        if self.continuous:
            self.train_loss_fn = get_sde_loss_fn(self.sde, True, reduce_mean=self.reduce_mean,
                                    continuous=True, likelihood_weighting=self.likelihood_weighting)
            self.eval_loss_fn = get_sde_loss_fn(self.sde, False, reduce_mean=self.reduce_mean,
                                    continuous=True, likelihood_weighting=self.likelihood_weighting)
        else:
            assert not self.likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
            if isinstance(self.sde, VESDE):
                self.train_loss_fn = get_smld_loss_fn(self.sde, True, reduce_mean=self.reduce_mean)
                self.eval_loss_fn = get_smld_loss_fn(self.sde, False, reduce_mean=self.reduce_mean)
            elif isinstance(self.sde, VPSDE):
                self.train_loss_fn = get_ddpm_loss_fn(self.sde, True, reduce_mean=self.reduce_mean)
                self.eval_loss_fn = get_ddpm_loss_fn(self.sde, False, reduce_mean=self.reduce_mean)
            else:
                raise ValueError(f"Discrete training for {self.sde.__class__.__name__} is not recommended.")

        # Building sampling functions
        if config.training.snapshot_sampling:
            sampling_shape = (config.training.batch_size, config.data.shape)
            self.sampling_fn = sampling.get_sampling_fn(config, self.sde, sampling_shape, self.sampling_eps)
    
    def training_step(self, batch, batch_idx):
        loss = self.train_loss_fn(self.score_model, batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.eval_loss_fn(self.score_model, batch)
        self.log('eval_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def sample(self, return_evolution=False):   
        if return_evolution:
            sample, evolution, times = self.sampling_fn(self.score_model, return_evolution=True)        
            return sample, evolution, times
        else:
            sample, n = self.sampling_fn(self.score_model)
            return sample

    def configure_optimizers(self):
        optimizer = losses.get_optimizer(self.config, self.score_model.parameters())
        return optimizer

    def load_sde(self, config):
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

