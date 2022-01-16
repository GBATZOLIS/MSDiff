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
        self.show_evolution = config.training.show_evolution

        #evaluation settings
        self.num_samples = config.eval.num_samples
        self.predictor = config.eval.predictor
        self.corrector = config.eval.corrector
        self.p_steps = config.eval.p_steps
        self.c_steps = config.eval.c_steps
        self.probability_flow = config.eval.probability_flow
        self.denoise = config.eval.denoise
        self.adaptive = config.eval.adaptive
        self.gamma = config.eval.gamma
        self.save_samples_dir = os.path.join(config.base_log_path, config.experiment_name, 'samples')
    
    def update_config(self, pl_module):
        pl_module.config = self.config

    def on_test_start(self, trainer, pl_module):
        self.update_config(pl_module)

    def generate_synthetic_dataset(self, pl_module, p_steps, adaptive, gamma):
        adaptive_name = 'KL-adaptive' if adaptive else 'uniform'
        eq = 'ode' if self.probability_flow else 'sde'
        p_step_dir = os.path.join(self.save_samples_dir, 'eq(%s)-p(%s)-c(%s)' % (eq, pl_module.config.eval.predictor, pl_module.config.eval.corrector), adaptive_name, '%.2f' % gamma, '%d' % p_steps)
        Path(p_step_dir).mkdir(parents=True, exist_ok=True)

        num_generated_samples=0
        while num_generated_samples < self.num_samples:
            samples, info = pl_module.sample(show_evolution=True,
                                          predictor=self.predictor,
                                          corrector=self.corrector,
                                          p_steps=p_steps,
                                          c_steps=self.c_steps,
                                          probability_flow=self.probability_flow,
                                          denoise=self.denoise,
                                          adaptive=adaptive,
                                          gamma=gamma)
            
            '''
            #saving code -> debug ddim with uniformly placed steps.
            num_generated_samples+=samples.size(0)
            evolution = info['evolution']
            for i in range(evolution.size(0)):
                p_step_dir_batch = os.path.join(p_step_dir, '%d' % num_generated_samples)
                Path(p_step_dir_batch).mkdir(parents=True, exist_ok=True)
                normalised_grid_evolution_step = torchvision.utils.make_grid(evolution[i], normalize=True, scale_each=True)
                fp = os.path.join(p_step_dir_batch, '%d.png' % (i+1))
                save_image(normalised_grid_evolution_step, fp)
            '''
            
            samples = torch.clamp(samples, min=0, max=1)

            batch_size = samples.size(0)
            if num_generated_samples+batch_size <= self.num_samples:
                for i in range(samples.size(0)):
                    fp = os.path.join(p_step_dir, '%d.png' % (num_generated_samples+i+1))
                    save_image(samples[i, :, :, :], fp)
            elif num_generated_samples+batch_size > self.num_samples:
                for i in range(self.num_samples - num_generated_samples): #add what is missing to fill the basket.
                    fp = os.path.join(p_step_dir, '%d.png' % (num_generated_samples+i+1))
                    save_image(samples[i, :, :, :], fp)

            num_generated_samples+=samples.size(0)
            
            
    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            for adaptive in self.adaptive:
                for gamma in self.gamma:
                    for p_steps in self.p_steps:
                        self.generate_synthetic_dataset(pl_module, p_steps, adaptive, gamma)

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = pl_module.current_epoch
        if current_epoch >= 2 and current_epoch % 5 == 0:
            samples, _ = pl_module.sample(num_samples=self.config.eval.batch_size)
            self.visualise_samples(samples, pl_module)

    def visualise_samples(self, samples, pl_module):
        # log sampled images
        sample_imgs =  samples.cpu()
        grid_images = torchvision.utils.make_grid(sample_imgs, normalize=True, scale_each=True)
        pl_module.logger.experiment.add_image('generated_images_%d' % pl_module.current_epoch, grid_images, pl_module.current_epoch)