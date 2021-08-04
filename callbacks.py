import torch
from pytorch_lightning.callbacks import Callback
from utils import plot, plot_line, compute_grad

class VisualisationCallback(Callback):

    def on_train_start(self, trainer, pl_module):
        # pl_module.logxger.log_hyperparams(params=pl_module.config.to_dict())
        self.visualise_samples(pl_module)

    def on_epoch_end(self,trainer, pl_module):
        if pl_module.current_epoch % 100 == 0:
            self.visualise_samples(pl_module)
            self.visualise_grad_norm(pl_module)

    
    def visualise_samples(self, pl_module):
        # log sampled images
        samples = pl_module.sample()
        samples_np =  samples.cpu().numpy()
        image = plot(samples_np[:,0],samples_np[:,1], 'samples epoch: ' + str(pl_module.current_epoch))
        pl_module.logger.experiment.add_image('samples', image, pl_module.current_epoch)

    def visualise_grad_norm(self, pl_module):
        # IMPROVE THIS: SAMPLE AT TIME T FOR GRADS AT TIME T
        # How to sample form p,t?
        samples = pl_module.sample()
        grad_norm_t =[]
        times=torch.linspace(0, pl_module.sde.N, 100 ,device=samples.device)
        for time in times:
            labels = time.repeat(samples.shape[0],*time.shape)
            gradients = compute_grad(f=pl_module.score_model, x=samples, t=labels)
            grad_norm = gradients.norm(2, dim=1).max().item()
            grad_norm_t.append(grad_norm)
        image = plot_line(times.cpu().numpy(),
                        grad_norm_t,
                        'Gradient Norms Epoch: ' + str(pl_module.current_epoch)
                        )
        pl_module.logger.experiment.add_image('grad_norms', image, pl_module.current_epoch)