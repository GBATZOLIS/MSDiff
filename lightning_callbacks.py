import torch
from pytorch_lightning.callbacks import Callback
from utils import scatter, plot, compute_grad, create_video
import torchvision
from torchvision.utils import make_grid
from . import utils
import numpy as np

_CALLBACKS = {}
def register_callback(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CALLBACKS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CALLBACKS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_callback_by_name(name):
    return _CALLBACKS[name]

def get_callbacks(visualization_callback, show_evolution):
    callbacks=[get_callback_by_name('ema')()]
    callbacks.append(get_callback_by_name(visualization_callback)(show_evolution=show_evolution))
    return callbacks

  
  
@register_callback(name='ema')
class EMACallback(Callback):

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        pl_module.ema.update(pl_module.score_model.parameters())

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        pl_module.ema.store(pl_module.score_model.parameters())
        pl_module.ema.copy_to(pl_module.score_model.parameters())

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.ema.restore(pl_module.score_model.parameters())

@register_callback(name='base')
class ImageVisualizationCallback(Callback):
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

        self.visualise_samples(samples, pl_module)

    def visualise_samples(self, samples, pl_module):
        # log sampled images
        sample_imgs =  samples.cpu()
        grid_images = torchvision.utils.make_grid(sample_imgs, normalize=True)
        pl_module.logger.experiment.add_image('generated_images', grid_images, pl_module.current_epoch)
    
    def visualise_evolution(self, evolution, pl_module):
        #to be implemented
        return



@register_callback(name='GradientVisualization')
class GradientVisualizer(Callback):

    def on_epoch_end(self,trainer, pl_module):
        if pl_module.current_epoch % 500 == 0:
            _, sampling_info = pl_module.sample(show_evolution=True)
            evolution, times = sampling_info['evolution'], sampling_info['times']
            self.visualise_grad_norm(evolution, times, pl_module)

    def visualise_grad_norm(self, evolution, times, pl_module):
        grad_norm_t =[]
        for i in range(evolution.shape[0]):
            t = times[i]
            samples = evolution[i]
            vec_t = torch.ones(times.shape[0], device=t.device) * t
            gradients = compute_grad(f=pl_module.score_model, x=samples, t=vec_t)
            grad_norm = gradients.norm(2, dim=1).max().item()
            grad_norm_t.append(grad_norm)
        image = plot(times.cpu().numpy(),
                        grad_norm_t,
                        'Gradient Norms Epoch: ' + str(pl_module.current_epoch)
                        )
        pl_module.logger.experiment.add_image('grad_norms', image, pl_module.current_epoch)

@register_callback(name='2DVisualization')
class TwoDimVizualizer(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.evolution = show_evolution

    def on_train_start(self, trainer, pl_module):
        # pl_module.logxger.log_hyperparams(params=pl_module.config.to_dict())
        samples, _ = pl_module.sample()
        self.visualise_samples(samples, pl_module)

    def on_epoch_end(self,trainer, pl_module):
        if pl_module.current_epoch % 500 == 0 \
            and pl_module.current_epoch % 2500 != 0:
            samples, _ = pl_module.sample()
            self.visualise_samples(samples, pl_module)
        if self.evolution and pl_module.current_epoch % 2500 == 0:
            samples, sampling_info = pl_module.sample(show_evolution=True)
            evolution = sampling_info['evolution']
            self.visualise_evolution(evolution, pl_module)

    def visualise_samples(self, samples, pl_module):
        # log sampled images
        samples_np =  samples.cpu().numpy()
        image = scatter(samples_np[:,0],samples_np[:,1], 
                        title='samples epoch: ' + str(pl_module.current_epoch))
        pl_module.logger.experiment.add_image('samples', image, pl_module.current_epoch)
    
    def visualise_evolution(self, evolution, pl_module):
        title = 'samples epoch: ' + str(pl_module.current_epoch)
        video_tensor = create_video(evolution, 
                                    title=title,
                                    xlim=[-1,1],
                                    ylim=[-1,1])
        tag='Evolution_epoch_%d' % pl_module.current_epoch
        pl_module.logger.experiment.add_video(tag=tag, vid_tensor=video_tensor, fps=video_tensor.size(1)//20)


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

@register_callback(name='haar_multiscale')
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

@register_callback(name='paired')
class PairedVisualizationCallback(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.show_evolution = show_evolution

    def on_epoch_end(self, trainer, pl_module):
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

    def visualise_paired_samples(self, y, x, pl_module, batch_idx):
        # log sampled images
        y_norm, x_norm = normalise_per_image(y).cpu(), normalise_per_image(x).cpu()
        concat_sample = torch.cat([y_norm, x_norm], dim=-1)
        grid_images = make_grid(concat_sample, nrow=int(np.sqrt(concat_sample.size(0))), normalize=False)
        pl_module.logger.experiment.add_image('generated_images_%d' % batch_idx, grid_images, pl_module.current_epoch)
    
    def visualise_evolution(self, evolution, pl_module):
        #to be implemented
        return