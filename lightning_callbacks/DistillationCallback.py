import torch
from pytorch_lightning.callbacks import Callback
from utils import scatter, plot, compute_grad, create_video
from torchvision.utils import make_grid, save_image
from models.ema import ExponentialMovingAverage
import torchvision
from . import utils
import numpy as np
import os
from pathlib import Path

@utils.register_callback(name='distillation')
class DistillationCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config

        #evaluation settings
        self.num_samples = config.eval.num_samples
    
    def update_config(self, pl_module):
        pl_module.config = self.config

    def on_test_start(self, trainer, pl_module):
        self.update_config(pl_module)

    def generate_synthetic_dataset(self, pl_module):
        save_dir = os.path.join(self.config.distillation.log_path, 'distillation_it_%d' % pl_module.iteration, 'samples')
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        num_generated_samples=0
        while num_generated_samples < self.num_samples:
            samples = pl_module.sample(num_samples=self.config.eval.batch_size)
            samples = torch.clamp(samples, min=0, max=1)

            batch_size = samples.size(0)
            if num_generated_samples+batch_size <= self.num_samples:
                for i in range(samples.size(0)):
                    fp = os.path.join(save_dir, '%d.png' % (num_generated_samples+i+1))
                    save_image(samples[i, :, :, :], fp)
            elif num_generated_samples+batch_size > self.num_samples:
                for i in range(self.num_samples - num_generated_samples): #add what is missing to fill the basket.
                    fp = os.path.join(save_dir, '%d.png' % (num_generated_samples+i+1))
                    save_image(samples[i, :, :, :], fp)

            num_generated_samples+=samples.size(0)
            
            
    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            self.generate_synthetic_dataset(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = pl_module.current_epoch
        if current_epoch != 0 and current_epoch % 5 == 0:
            samples = pl_module.sample(num_samples=self.config.eval.batch_size)
            self.visualise_samples(samples, pl_module)

    def visualise_samples(self, samples, pl_module):
        # log sampled images
        sample_imgs =  samples.cpu()
        grid_images = torchvision.utils.make_grid(sample_imgs, normalize=True, scale_each=True)
        pl_module.logger.experiment.add_image('generated_images_%d' % pl_module.current_epoch, grid_images, pl_module.current_epoch)