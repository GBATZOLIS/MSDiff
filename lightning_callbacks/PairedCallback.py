from . import utils
import torch
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid
import numpy as np


def normalise(c, value_range=None):
    x = c.clone()
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
    return normalised_evolution

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
        if current_epoch == 0 or current_epoch % 10 != 5:
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
                print('sde_y sigma_max: %.5f ' % pl_module.sde['y'].sigma_max)
                conditional_samples, sampling_info = pl_module.sample(y.to(pl_module.device), show_evolution=True)
                evolution = sampling_info['evolution']
                self.visualise_evolution(evolution, pl_module, tag='val_joint_evolution_batch_%d_epoch_%d' % (i, current_epoch))
            else:
                conditional_samples, _ = pl_module.sample(y.to(pl_module.device), show_evolution=False)

            self.visualise_paired_samples(y, conditional_samples, pl_module, i+1)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        y, x = batch
        print('sde_y sigma_max: %.5f ' % pl_module.sde['y'].sigma_max)
        samples, sampling_info = pl_module.sample(y.to(pl_module.device), show_evolution=True) #sample x conditioned on y
        evolution = sampling_info['evolution']
        #self.visualise_paired_samples(y, samples, pl_module, batch_idx, phase='test')
        self.visualise_evolution(evolution, pl_module, tag='test_joint_evolution_batch_%d' % batch_idx)

    def visualise_paired_samples(self, y, x, pl_module, batch_idx, phase='train'):
        # log sampled images
        y_norm, x_norm = normalise_per_image(y).cpu(), normalise_per_image(x).cpu()
        concat_sample = torch.cat([y_norm, x_norm], dim=-1)
        grid_images = make_grid(concat_sample, nrow=int(np.sqrt(concat_sample.size(0))), normalize=False)
        pl_module.logger.experiment.add_image('generated_images_%sbatch_%d' % (phase, batch_idx), grid_images, pl_module.current_epoch)
    
    def visualise_evolution(self, evolution, pl_module, tag):
        norm_evolution_x = normalise_evolution(evolution['x'])
        norm_evolution_y = normalise_evolution(evolution['y'])
        joint_evolution = torch.cat([norm_evolution_y, norm_evolution_x], dim=-1)
        video_grid = create_video_grid(joint_evolution)
        pl_module.logger.experiment.add_video(tag, video_grid.unsqueeze(0), fps=50)

@utils.register_callback(name='paired3D')
class PairedVisualizationCallback(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.show_evolution = show_evolution

    def convert_to_3D(self, x):
        if len(x.shape[1:]) == 3:
            x = torch.swapaxes(x, 1, -1).unsqueeze(1)
            print(x.size())
            return x
        elif len(x.shape[1:]) == 4:
            return x
        else:
            raise NotImplementedError('x dimensionality is not supported.')

    def generate_paired_video(self, pl_module, Y, I, cond_samples, dim, batch_idx):
        #dim: the sliced dimension (choices: 1,2,3)
        B = Y.size(0)

        if cond_samples is not None:
            raw_length = 1+cond_samples.size(0)+1
        else:
            raw_length = 2

        frames = Y.size(dim+1)
        video_grid = []
        for frame in range(frames):
            if dim==1:
                dim_cut = torch.zeros(tuple([B*raw_length, 1, I.shape[3], I.shape[4]])).type_as(Y)
            elif dim==2:
                dim_cut = torch.zeros(tuple([B*raw_length, 1, I.shape[2], I.shape[4]])).type_as(Y)
            elif dim==3:
                dim_cut = torch.zeros(tuple([B*raw_length, 1, I.shape[2], I.shape[3]])).type_as(Y)

            for i in range(B):
                if dim==1:
                    dim_cut[i*raw_length] = normalise(Y[i, 0, frame, :, :]).unsqueeze(0)
                    dim_cut[(i+1)*raw_length-1] = normalise(I[i, 0, frame, :, :]).unsqueeze(0)
                    if cond_samples is not None:
                        for j in range(cond_samples.size(0)):
                            dim_cut[i*raw_length+j+1] = normalise(cond_samples[j, i, 0, frame, :, :]).unsqueeze(0)
                elif dim==2:
                    dim_cut[i*raw_length] = normalise(Y[i, 0, :, frame, :]).unsqueeze(0)
                    dim_cut[(i+1)*raw_length-1] = normalise(I[i, 0, :, frame, :]).unsqueeze(0)
                    if cond_samples is not None:
                        for j in range(cond_samples.size(0)):
                            dim_cut[i*raw_length+j+1] = normalise(cond_samples[j, i, 0, :, frame, :]).unsqueeze(0)
                elif dim==3:
                    dim_cut[i*raw_length] = normalise(Y[i, 0, :, :, frame]).unsqueeze(0)
                    dim_cut[(i+1)*raw_length-1] = normalise(I[i, 0, :, :, frame]).unsqueeze(0)
                    if cond_samples is not None:
                        for j in range(cond_samples.size(0)):
                            dim_cut[i*raw_length+j+1] = normalise(cond_samples[j, i, 0, :, :, frame]).unsqueeze(0)

            grid_cut = make_grid(tensor=dim_cut, nrow=raw_length, normalize=False)
            video_grid.append(grid_cut)

        video_grid = torch.stack(video_grid, dim=0).unsqueeze(0)
        #print(video_grid.size())

        str_title = 'paired_video_epoch_%d_batch_%d_dim_%d' % (pl_module.current_epoch, batch_idx, dim)
        pl_module.logger.experiment.add_video(str_title, video_grid, pl_module.current_epoch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        current_epoch = pl_module.current_epoch
        if batch_idx!=2 or current_epoch == 0 or current_epoch % 50 != 0:
            return
        
        y, x = batch        
        cond_samples, _ = pl_module.sample(y.to(pl_module.device), show_evolution=self.show_evolution)
        val_rec_loss = torch.mean(torch.abs(x.to(pl_module.device)-cond_samples))
        pl_module.logger.experiment.add_scalar('val_rec_loss_epoch_%d_batch_%d' % (current_epoch, batch_idx), val_rec_loss)

        x = self.convert_to_3D(x).cpu()
        cond_samples = self.convert_to_3D(cond_samples).unsqueeze(0).cpu()
        y = self.convert_to_3D(y).cpu()
        

        for dim in [1, 2, 3]:
            self.generate_paired_video(pl_module, y, x, cond_samples, dim, batch_idx)