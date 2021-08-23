from . import utils
import torch
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid
import numpy as np

def normalise_per_image(x, value_range=None):
    for i in range(x.size(0)):
        x[i,::] = normalise(x[i,::], value_range=value_range)
    return x

def permute_channels(haar_image, forward=True):
        permuted_image = torch.zeros_like(haar_image)
        if forward:
            for i in range(4):
                if i == 0:
                    k = 1
                elif i == 1:
                    k = 0
                else:
                    k = i
                for j in range(3):
                    permuted_image[:, 3*k+j, :, :] = haar_image[:, 4*j+i, :, :]
        else:
            for i in range(4):
                if i == 0:
                    k = 1
                elif i == 1:
                    k = 0
                else:
                    k = i
                
                for j in range(3):
                    permuted_image[:,4*j+k,:,:] = haar_image[:, 3*i+j, :, :]

        return permuted_image

def normalise(x, value_range=None):
    if value_range is None:
        x -= x.min()
        x /= x.max()
    else:
        x -= value_range[0]
        x /= value_range[1]
    return x

def normalise_per_band(permuted_haar_image):
    normalised_image = permuted_haar_image.clone()
    for i in range(4):
        normalised_image[:, 3*i:3*(i+1), :, :] = normalise(permuted_haar_image[:, 3*i:3*(i+1), :, :])
    return normalised_image #normalised permuted haar transformed image

def create_supergrid(normalised_permuted_haar_images):
    haar_super_grid = []
    for i in range(normalised_permuted_haar_images.size(0)):
        shape = normalised_permuted_haar_images[i].shape
        haar_grid = make_grid(normalised_permuted_haar_images[i].reshape((-1, 3, shape[1], shape[2])), nrow=2)
        haar_super_grid.append(haar_grid)
    
    super_grid = make_grid(haar_super_grid, nrow=int(np.sqrt(normalised_permuted_haar_images.size(0))))
    return super_grid

@utils.register_callback(name='haar_multiscale')
class HaarMultiScaleVisualizationCallback(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.show_evolution = show_evolution

    def on_epoch_end(self, trainer, pl_module):
        if self.show_evolution:
            samples, sampling_info = pl_module.sample(show_evolution=True)
            evolution = sampling_info['evolution']
            self.visualise_evolution(evolution, pl_module)
        else:
            samples, _ = pl_module.sample(show_evolution=False)
            normalised_samples = normalise_per_band(samples)
            haar_grid = create_supergrid(normalised_samples)
            pl_module.logger.experiment.add_image('haar_supergrid', haar_grid, pl_module.current_epoch)

            back_permuted_samples = permute_channels(samples, forward=False)
            image_grid = pl_module.haar_transform.inverse(back_permuted_samples)
            image_grid = make_grid(normalise_per_image(image_grid), nrow=int(np.sqrt(image_grid.size(0))))
            pl_module.logger.experiment.add_image('image_grid', image_grid, pl_module.current_epoch)

    def visualise_evolution(self, evolution, pl_module):
        haar_super_grid_evolution = []
        for i in range(evolution.size(0)):
            haar_super_grid_evolution.append(create_supergrid(normalise_per_band(evolution[i])))
        haar_super_grid_evolution = torch.stack(haar_super_grid_evolution).unsqueeze(0)
        pl_module.logger.experiment.add_video('haar_super_grid_evolution', haar_super_grid_evolution, pl_module.current_epoch, fps=50)