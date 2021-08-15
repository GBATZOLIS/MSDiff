import torch
from pytorch_lightning.callbacks import Callback
from utils import scatter, plot, compute_grad, create_video
from pytorch_lightning.callbacks import Callback
from models.ema import ExponentialMovingAverage
import torchvision

class EMACallback(Callback):

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        pl_module.ema.update(pl_module.score_model.parameters())

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        pl_module.ema.store(pl_module.score_model.parameters())
        pl_module.ema.copy_to(pl_module.score_model.parameters())

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.ema.restore(pl_module.score_model.parameters())


class ImageVisulaizationCallback(Callback):
     
    #def on_train_start(self, trainer, pl_module):
        # pl_module.logxger.log_hyperparams(params=pl_module.config.to_dict())
        #samples = pl_module.sample()
        #self.visualise_samples(samples, pl_module)
         
    
    def on_epoch_end(self,trainer, pl_module):
        samples = pl_module.sample()
        self.visualise_samples(samples, pl_module)

    
    def visualise_samples(self, samples, pl_module):
        # log sampled images
        sample_imgs =  samples.cpu()
        unnormalised_imgs = sample_imgs / 2 + 0.5 # improve this line
        grid_images = torchvision.utils.make_grid(unnormalised_imgs)
        pl_module.logger.experiment.add_image('generated_images', grid_images, pl_module.current_epoch)



class VisualisationCallback(Callback):

    def on_train_start(self, trainer, pl_module):
        # pl_module.logxger.log_hyperparams(params=pl_module.config.to_dict())
        samples = pl_module.sample()
        self.visualise_samples(samples, pl_module)

    def on_epoch_end(self,trainer, pl_module):
        if pl_module.current_epoch % 500 == 0:
            samples, evolution, times = pl_module.sample(return_evolution=True)
            self.visualise_samples(samples, pl_module)
            self.visualise_grad_norm(evolution, times, pl_module)
        if pl_module.current_epoch % 2500 == 0:
            self.visualise_evolution(evolution, pl_module)

    
    def visualise_samples(self, samples, pl_module):
        # log sampled images
        samples_np =  samples.cpu().numpy()
        image = scatter(samples_np[:,0],samples_np[:,1], 
                        title='samples epoch: ' + str(pl_module.current_epoch))
        pl_module.logger.experiment.add_image('samples', image, pl_module.current_epoch)

    def visualise_grad_norm(self, evolution, times, pl_module):
        grad_norm_t =[]
        for i in range(evolution.shape[0]):
            t = times[i]
            samples=evolution[i]
            vec_t = torch.ones(times.shape[0], device=t.device) * t
            gradients = compute_grad(f=pl_module.score_model, x=samples, t=vec_t)
            grad_norm = gradients.norm(2, dim=1).max().item()
            grad_norm_t.append(grad_norm)
        image = plot(times.cpu().numpy(),
                        grad_norm_t,
                        'Gradient Norms Epoch: ' + str(pl_module.current_epoch)
                        )
        pl_module.logger.experiment.add_image('grad_norms', image, pl_module.current_epoch)

    
    def visualise_evolution(self, evolution, pl_module):
        title = 'samples epoch: ' + str(pl_module.current_epoch)
        video_tensor = create_video(evolution, 
                                    title=title,
                                    xlim=[-1,1],
                                    ylim=[-1,1])
        tag='Evolution_epoch_%d' % pl_module.current_epoch
        pl_module.logger.experiment.add_video(tag=tag, vid_tensor=video_tensor, fps=video_tensor.size(1)//20)

        # title = 'samples epoch: ' + str(pl_module.current_epoch)
        # video_tensor = create_video(evolution, 
        #                             title=title,
        #                             xlim=[-1,1],
        #                             ylim=[-1,1])
        # tag='Zoom_evolution_epoch_%d' % pl_module.current_epoch
        # pl_module.logger.experiment.add_video(tag=tag, vid_tensor=video_tensor, fps=video_tensor.size(1)//20)

    