import losses_lightning
import losses
import pytorch_lightning as pl
import sde_lib
import sampling
from models.ema import ExponentialMovingAverage
from models import utils as mutils
from utils import plot
from models import ddpm, ncsnv2, fcn

class SdeGenerativeModel(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # Initialize model
        self.config = config
        self.score_model = mutils.create_model(config)
        self.ema = ExponentialMovingAverage(self.score_model.parameters(), decay=config.model.ema_rate)

        # Setup SDEs
        if config.training.sde.lower() == 'vpsde':
            sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            sampling_eps = 1e-3
        elif config.training.sde.lower() == 'subvpsde':
            sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
            sampling_eps = 1e-3
        elif config.training.sde.lower() == 'vesde':
            sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
            sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")

        # Set up loss functions
        # Build one-step training and evaluation functions
        continuous = config.training.continuous
        reduce_mean = config.training.reduce_mean
        likelihood_weighting = config.training.likelihood_weighting
        self.train_step_fn = losses_lightning.get_step_fn(sde, train=True, 
                                            reduce_mean=reduce_mean, continuous=continuous,
                                            likelihood_weighting=likelihood_weighting)

        
        self.eval_step_fn = losses_lightning.get_step_fn(sde, train=False, 
                                            reduce_mean=reduce_mean, continuous=continuous,
                                            likelihood_weighting=likelihood_weighting)
        # Building sampling functions
        if config.training.snapshot_sampling:
            sampling_shape = (config.training.batch_size, config.data.dim)
            self.sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, sampling_eps)
    
    def training_step(self, batch, batch_idx):
        loss = self.train_step_fn(self.score_model, batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    #def validation_step(self, batch, batch_idx):
        #batch=batch[0]
        #return self.eval_step_fn(self.score_model, batch)
    
    def sample(self):
        sample, n = self.sampling_fn(self.score_model)
        return sample        

    def configure_optimizers(self):
        optimizer = losses_lightning.get_optimizer(self.config, self.score_model.parameters())
        return optimizer


