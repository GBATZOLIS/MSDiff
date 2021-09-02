from . import utils
import torch
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid
import numpy as np


def normalise(x, value_range=None):
    if value_range is None:
        x -= x.min()
        x /= x.max()
    else:
        x -= value_range[0]
        x /= value_range[1]
    return x

def normalise_per_image(x, value_range=None):
    for i in range(x.size(0)):
        x[i,::] = normalise(x[i,::], value_range=value_range)
    return x

def normalise_evolution(evolution):
    normalised_evolution = torch.ones_like(evolution)
    for i in range(evolution.size(0)):
        normalised_evolution[i] = normalise_per_image(evolution[i])
    return normalise_evolution

def create_video_grid(evolution):
    video_grid = []
    for i in range(evolution.size(0)):
        video_grid.append(make_grid(evolution[i], nrow=int(np.sqrt(evolution[i].size(0))), normalize=False))
    return torch.stack(video_grid)

@utils.register_callback(name='paired')
class PairedVisualizationCallback(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.show_evolution = show_evolution

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = pl_module.current_epoch
        if current_epoch == 0 or current_epoch % 5 != 0:
            return 
        
        dataloader_iterator = iter(trainer.datamodule.val_dataloader())
        num_batches = 1
        for i in range(num_batches):
            try:
                y, x = next(dataloader_iterator)
            except StopIteration:
                print('Requested number of batches exceeds the number of batches available in the val dataloader.')
                break

            if self.show_evolution:
                conditional_samples, sampling_info = pl_module.sample(y.to(pl_module.device), show_evolution=True)
                evolution = sampling_info['evolution']
                self.visualise_evolution(evolution, pl_module)
            else:
                conditional_samples, _ = pl_module.sample(y.to(pl_module.device), show_evolution=False)

            self.visualise_paired_samples(y, conditional_samples, pl_module, i+1)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        y, x = batch
        _, sampling_info = pl_module.sample(y, show_evolution=True) #sample x conditioned on y
        evolution = sampling_info['evolution']
        self.visualise_evolution(evolution, pl_module, batch_idx)

    def visualise_paired_samples(self, y, x, pl_module, batch_idx):
        # log sampled images
        y_norm, x_norm = normalise_per_image(y).cpu(), normalise_per_image(x).cpu()
        concat_sample = torch.cat([y_norm, x_norm], dim=-1)
        grid_images = make_grid(concat_sample, nrow=int(np.sqrt(concat_sample.size(0))), normalize=False)
        pl_module.logger.experiment.add_image('generated_images_batch_%d' % batch_idx, grid_images, pl_module.current_epoch)
    
    def visualise_evolution(self, evolution, pl_module, batch_idx):
        norm_evolution_x = normalise_evolution(evolution['x'])
        norm_evolution_y = normalise_evolution(evolution['y'])
        joint_evolution = torch.cat([norm_evolution_y, norm_evolution_x], dim=-1)
        video_grid = create_video_grid(joint_evolution)
        pl_module.logger.experiment.add_video('joint_evolution_batch_%d' % batch_idx, video_grid, fps=50)