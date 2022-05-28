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
import pickle
from fast_sampling.computation_utils import get_adaptive_discretisation_fn
from lightning_callbacks.HaarMultiScaleCallback import normalise_per_image, permute_channels, normalise, normalise_per_band, create_supergrid

@utils.register_callback(name='configuration')
class ConfigurationSetterCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        # Configure SDE
        pl_module.configure_sde(pl_module.config)
        
        # Configure trainining and validation loss functions.
        pl_module.train_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=True)
        pl_module.eval_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=False)
    
    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.configure_sde(pl_module.config)


@utils.register_callback(name='decreasing_variance_configuration')
class DecreasingVarianceConfigurationSetterCallback(ConfigurationSetterCallback):
    def __init__(self, config):
        super().__init__()
        self.sigma_max_y_fn = get_reduction_fn(y0=config.model.sigma_max_y, 
                                               xk=config.model.reach_target_steps, 
                                               yk=config.model.sigma_max_y_target)
        
        self.sigma_min_y_fn = get_reduction_fn(y0=config.model.sigma_min_y, 
                                               xk=config.model.reach_target_steps, 
                                               yk=config.model.sigma_min_y_target)


    def on_fit_start(self, trainer, pl_module):
        # Configure SDE
        pl_module.configure_sde(pl_module.config)
        
        # Configure trainining and validation loss functions.
        pl_module.train_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=True)
        pl_module.eval_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=False)

    def reconfigure_conditioning_sde(self, trainer, pl_module):
        #calculate current sigma_max_y and sigma_min_y
        current_sigma_max_y = self.sigma_max_y_fn(pl_module.global_step)
        current_sigma_min_y = self.sigma_min_y_fn(pl_module.global_step)

        # Reconfigure SDE
        pl_module.reconfigure_conditioning_sde(pl_module.config, current_sigma_min_y, current_sigma_max_y)
        
        # Reconfigure trainining and validation loss functions. -  we might not need to reconfigure the losses.
        pl_module.train_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=True)
        pl_module.eval_loss_fn = pl_module.configure_loss_fn(pl_module.config, train=False)
        
        return current_sigma_min_y, current_sigma_max_y

    def on_sanity_check_start(self, trainer, pl_module):
        self.reconfigure_conditioning_sde(trainer, pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        current_sigma_min_y, current_sigma_max_y = self.reconfigure_conditioning_sde(trainer, pl_module)
        pl_module.sigma_max_y = torch.tensor(current_sigma_max_y).float()
        pl_module.sigma_min_y = torch.tensor(current_sigma_min_y).float()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        current_sigma_min_y, current_sigma_max_y = self.reconfigure_conditioning_sde(trainer, pl_module)
        
        pl_module.sigma_max_y = torch.tensor(current_sigma_max_y).float()
        pl_module.logger.experiment.add_scalar('sigma_max_y', current_sigma_max_y, pl_module.global_step)
        
        pl_module.sigma_min_y = torch.tensor(current_sigma_min_y).float()
        pl_module.logger.experiment.add_scalar('sigma_min_y', current_sigma_min_y, pl_module.global_step)

    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.configure_sde(config = pl_module.config, 
                                sigma_min_y = pl_module.sigma_min_y,
                                sigma_max_y = pl_module.sigma_max_y)


def get_reduction_fn(y0, xk, yk):
    #get the reduction function that starts at y0 and reaches point yk at xk steps.
    #the function follows an inverse multiplicative rate.
    def f(x):
        return xk*yk*y0/(x*(y0-yk)+xk*yk)
    return f

def get_deprecated_sigma_max_y_fn(reduction, reach_target_in_epochs, starting_transition_iterations):
    if reduction == 'linear':
        def sigma_max_y(global_step, current_epoch, start_value, target_value):
            if current_epoch >= reach_target_in_epochs:
                current_sigma_max_y = target_value
            else:
                current_sigma_max_y = start_value - current_epoch/reach_target_in_epochs*(start_value - target_value)

            return current_sigma_max_y
                
    elif reduction == 'inverse_exponentional':
        def sigma_max_y(global_step, current_epoch, start_value, target_value):
            x_prev = 0
            x_next = starting_transition_iterations
            x_add = starting_transition_iterations

            while global_step > x_next:
                x_add *= 2
                x_prev = x_next
                x_next = x_add + x_prev
                start_value = start_value/2

            target_value = start_value/2
            current_sigma_max_y = start_value - (global_step-x_prev)/(x_next-x_prev)*(start_value - target_value)
            return current_sigma_max_y
    else:
        raise NotImplementedError('Reduction type %s is not supported yet.' % reduction)

    return sigma_max_y

@utils.register_callback(name='multiscale_base')
class MultiscaleImageVisualizationCallback(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config

        #evaluation configuration
        self.num_samples = config.eval.num_samples
        self.predictor = config.eval.predictor
        self.corrector = config.eval.corrector
        self.p_steps = config.eval.p_steps
        self.c_steps = config.eval.c_steps
        self.probability_flow = config.eval.probability_flow
        self.denoise = config.eval.denoise

        if hasattr(config.eval, 'checkpoint_iteration'):
            if config.eval.checkpoint_iteration is not None:
                self.save_samples_dir = os.path.join(config.base_log_path, config.experiment_name, 'samples', '%d' % config.eval.checkpoint_iteration)
            else:
                self.save_samples_dir = os.path.join(config.base_log_path, config.experiment_name, 'samples')
        else:
            self.save_samples_dir = os.path.join(config.base_log_path, config.experiment_name, 'samples')

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            for predictor in self.predictor:
                for p_steps in self.p_steps:
                    self.generate_dataset(pl_module, predictor, p_steps)
            

            #commented section to be permanently deleted in the future. It was done for the purposes of creation of a demonstration.
            '''
            #get the perturbed data
            pertub_data_evolution = self.get_converged_evolution(pl_module, batch)

            #convert it to perturbed super-grids
            super_grid_evolution = {}
            for T2 in pertub_data_evolution.keys():
                super_grid = self.convert_to_haar_grid_space(pertub_data_evolution[T2], pl_module)
                super_grid_evolution[T2] = super_grid[0] #keep only the first image
            
            #save the super grids for different timepoints
            base_dir = os.path.join(self.save_samples_dir, 'teaser')
            Path(base_dir).mkdir(parents=True, exist_ok=True)
            for T2 in pertub_data_evolution.keys():
                fp = os.path.join(base_dir, '%.2f.png' % T2)
                save_image(super_grid_evolution[T2], fp)
            '''

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = pl_module.current_epoch
        if current_epoch % 5 == 0 and current_epoch != 0:
            for predictor in ['reverse_diffusion']:
                psteps_per_scale = 1000//pl_module.num_scales
                samples, sampling_info = pl_module.sample(num_samples=pl_module.config['d1'].eval.batch_size, 
                                                          p_steps=psteps_per_scale, predictor=predictor)
                self.visualise_samples(samples, pl_module, steps_per_scale=psteps_per_scale, predictor=predictor)
            
    def visualise_samples(self, samples, pl_module, steps_per_scale, predictor):
        # log sampled images
        sample_imgs =  samples.cpu()
        grid_images = torchvision.utils.make_grid(sample_imgs, normalize=True, scale_each=True)
        pl_module.logger.experiment.add_image('generated_samples_%d_psteps_per_scale_%d_predictor_%s' % (pl_module.global_step, steps_per_scale, predictor), grid_images, pl_module.current_epoch)
    
    def detect_haar_depth(self, haar_x : dict):
        for key in haar_x.keys():
            if key.startswith('a'):
                approx_key = key
                break
        return int(approx_key[1])

    def convert_to_haar_grid_space(self, haar_x, pl_module):
        depth = self.detect_haar_depth(haar_x)

        super_grids = []
        for j in range(haar_x['a%d' % depth].size(0)):
            a = normalise(haar_x['a%d' % depth][j, ::])
            for i in range(depth):
                d = normalise(haar_x['d%d'%(depth-i)][j, ::])
                concat = torch.cat((a,d), dim=0)
                shape = concat.shape
                a = make_grid(concat.reshape((-1, 3, shape[1], shape[2])), nrow=2, padding=0) 
            super_grids.append(a)

        super_grid = torch.stack(super_grids)
        return super_grid

    def get_converged_evolution(self, pl_module, batch):
        batch = pl_module.convert_to_haar_space(batch, max_depth=pl_module.num_scales-1)

        pertub_data_evolution = {}
        pertub_data_evolution[0.] = batch
        for scale_idx in range(1, pl_module.num_scales + 1):
            T1, T2 = pl_module.compute_interval(scale_idx)
            pertub_data_evolution[T2] = {}
            for member_name in batch.keys():
                z = torch.randn_like(batch[member_name])
                mean, std = pl_module.sde[member_name].marginal_prob(batch[member_name], torch.tensor(T2).type_as(batch[member_name]))
                perturbed_member = mean + std[(...,) + (None,) * len(batch[member_name].shape[1:])] * z
                pertub_data_evolution[T2][member_name] = perturbed_member
        return pertub_data_evolution

    def generate_dataset(self, pl_module, predictor, p_steps):
        def get_image_count(directory, img_format='png'):  
            initial_count = 0
            for path in os.listdir(directory):
                if os.path.isfile(os.path.join(directory, path)) and path.split('.')[-1]==img_format:
                    initial_count += 1
            return initial_count

        eq = 'ode' if self.probability_flow else 'sde'

        p_step_dir = os.path.join(self.save_samples_dir, 'eq(%s)-p(%s)-c(%s)' % (eq, predictor, self.config.eval.corrector), '%d' % p_steps)
        Path(p_step_dir).mkdir(parents=True, exist_ok=True)

        num_generated_samples = get_image_count(p_step_dir)
        while num_generated_samples < self.num_samples:
            psteps_per_scale = p_steps // pl_module.num_scales
            samples, info = pl_module.sample(num_samples=self.config.eval.batch_size,
                                            predictor=predictor,
                                            corrector=self.corrector,
                                            p_steps=psteps_per_scale,
                                            c_steps=self.c_steps,
                                            probability_flow=self.probability_flow,
                                            denoise=self.denoise)
            if num_generated_samples == num_generated_samples + samples.size(0):
                timing_information = {}
                for scale_name in info.keys():
                    scale_time = info[scale_name]['time']
                    timing_information[scale_name] = scale_time
                
                f = open(os.path.join(p_step_dir, 'timing_information.pkl'), "wb")
                pickle.dump(timing_information, f)
                f.close()

            #grid_images = torchvision.utils.make_grid(samples.to('cpu'), normalize=True, scale_each=True)
            #fp = os.path.join(p_step_dir, 'grid.png')
            #save_image(grid_images, fp)

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


@utils.register_callback(name='base')
class ImageVisualizationCallback(Callback):
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
        self.adaptive_method = config.eval.adaptive_method
        self.gamma = config.eval.gamma
        self.alpha = config.eval.alpha
        self.starting_T = config.eval.starting_T
        #self.save_samples_dir = os.path.join(config.base_log_path, config.experiment_name, 'samples')

        if hasattr(config.eval, 'checkpoint_iteration'):
            if config.eval.checkpoint_iteration is not None:
                self.save_samples_dir = os.path.join(config.base_log_path, config.experiment_name, 'samples', '%d' % config.eval.checkpoint_iteration)
            else:
                self.save_samples_dir = os.path.join(config.base_log_path, config.experiment_name, 'samples')
        else:
            self.save_samples_dir = os.path.join(config.base_log_path, config.experiment_name, 'samples')

    def update_config(self, pl_module):
        pl_module.config = self.config

    def on_test_start(self, trainer, pl_module):
        self.update_config(pl_module)

    def generate_dataset(self, pl_module, predictor, p_steps):
        eq = 'ode' if self.probability_flow else 'sde'

        p_step_dir = os.path.join(self.save_samples_dir, 'eq(%s)-p(%s)-c(%s)' % (eq, predictor, pl_module.config.eval.corrector), '%d' % p_steps)
        Path(p_step_dir).mkdir(parents=True, exist_ok=True)

        num_generated_samples=0
        while num_generated_samples < self.num_samples:
            samples, info = pl_module.sample(predictor=predictor,
                                            corrector=self.corrector,
                                            p_steps=p_steps,
                                            c_steps=self.c_steps,
                                            probability_flow=self.probability_flow,
                                            denoise=self.denoise)
            
            if num_generated_samples==0:
                timing_information = {'sampling_time': info['time']}
                f = open(os.path.join(p_step_dir, 'timing_information.pkl'), "wb")
                pickle.dump(timing_information, f)
                f.close()

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
            for predictor in self.predictor:
                for p_steps in self.p_steps:
                    self.generate_dataset(pl_module, predictor, p_steps)

    def on_validation_epoch_end(self, trainer, pl_module):
        current_epoch = pl_module.current_epoch
        if current_epoch % 5 == 0 and current_epoch!=0:
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
        grid_images = torchvision.utils.make_grid(sample_imgs, normalize=True, scale_each=True)
        pl_module.logger.experiment.add_image('generated_images_%d' % pl_module.current_epoch, grid_images, pl_module.current_epoch)
    
    def visualise_evolution(self, evolution, pl_module):
        #to be implemented - has already been implemented for the conditional case
        return NotImplementedError('Visualisation of evolution not supported yet for unconditional sampling.')
    


@utils.register_callback(name='GradientVisualization')
class GradientVisualizer(Callback):

    def on_validation_epoch_end(self,trainer, pl_module):
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
    def __init__(self, config):
        super().__init__()
        self.evolution = config.training.show_evolution

    def on_train_start(self, trainer, pl_module):
        # pl_module.logxger.log_hyperparams(params=pl_module.config.to_dict())
        samples, _ = pl_module.sample()
        self.visualise_samples(samples, pl_module)

    def on_validation_epoch_end(self,trainer, pl_module):
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


