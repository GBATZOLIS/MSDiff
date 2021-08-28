import torch
from pytorch_lightning.callbacks import Callback
from utils import scatter, plot, compute_grad, create_video
import torchvision
from . import utils
import numpy as np

@utils.register_callback(name='configuration')
class ConfigurationSetterCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        # Configure SDE
        pl_module.configure_sde(pl_module.config)
        
        # Configure trainining and validation loss functions.
        pl_module.train_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=True)
        pl_module.eval_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=False)

@utils.register_callback(name='ema')
class EMACallback(Callback):

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        pl_module.ema.update(pl_module.score_model.parameters())

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        pl_module.ema.store(pl_module.score_model.parameters())
        pl_module.ema.copy_to(pl_module.score_model.parameters())

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.ema.restore(pl_module.score_model.parameters())

@utils.register_callback(name='base')
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



@utils.register_callback(name='GradientVisualization')
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

@utils.register_callback(name='2DVisualization')
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


