import losses
from losses import get_sde_loss_fn, get_smld_loss_fn, get_ddpm_loss_fn, get_general_sde_loss_fn
import pytorch_lightning as pl
import sde_lib
from sampling.unconditional import get_sampling_fn
from models import utils as mutils
from sde_lib import VESDE, VPSDE
from . import utils
import torch.optim as optim
import os
import torch
from fast_sampling.computation_utils import get_adaptive_discretisation_fn
import pickle 
import numpy as np
from timeit import default_timer as timer
import torch.nn as nn
from iunets.layers import InvertibleDownsampling2D

@utils.register_lightning_module(name='multiscale_base')
class MultiScaleSdeGenerativeModel(pl.LightningModule):
    def __init__(self, configs, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.config = {}

        # Initialize the score model
        self.score_model = nn.ModuleDict()

        self.num_scales = 0
        for config_name, config in configs.items():
            self.score_model[config.data.scale_name] = mutils.create_model(config)
            self.config[config.data.scale_name] = config
            self.num_scales += 1

        self.scale_name_to_index = self.get_scale_name_to_index()
        self.index_to_scale_name = self.get_index_to_scale_name()

        self.reverse_scale_order = self.get_scale_order(direction='reverse')
        self.forward_scale_order = self.get_scale_order(direction='forward')

    def get_scale_name_to_index(self, ):
        forward_scale_order = self.get_scale_order(direction='forward')
        scale_name_to_index = {}
        for i, scale_name in enumerate(forward_scale_order):
            scale_name_to_index[scale_name] = i+1
        return scale_name_to_index
    
    def get_index_to_scale_name(self, ):
        forward_scale_order = self.get_scale_order(direction='forward')
        index_to_scale_name = {}
        for i, scale_name in enumerate(forward_scale_order):
            index_to_scale_name[i+1] = scale_name
        return index_to_scale_name

    def get_scale_order(self, direction='reverse'):
        #this function needs modification to support different splitting strategies into scales
        #direction: [reverse, forward]

        forward_order = []
        for scale in range(1, self.num_scales):
            forward_order.append('d%d' % scale)

            if scale == self.num_scales-1:
                forward_order.append('a%d' % scale)

        if direction == 'reverse':
            return forward_order[::-1]
        elif direction == 'forward':
            return forward_order
        else:
            return NotImplementedError('direction can only be reverse or forward.')

    def configure_sde(self, configs):
        # Setup SDEs for every configuration
        self.sde = {}
        self.sampling_eps = 1e-3
        for config_name in configs.keys():
            config = configs[config_name]
            scale_name = config.data.scale_name
            self.sde[scale_name] = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    
    def configure_loss_fn(self, configs, train):
        loss_fn = get_general_sde_loss_fn(self.sde, train, multiscale=True, reduce_mean=True, continuous=True, likelihood_weighting=True)
        return loss_fn

    def compute_interval(self, scale_index):
        if scale_index == 1:
            return self.sampling_eps, 1/self.num_scales
        else:
            return (scale_index-1)/self.num_scales, scale_index/self.num_scales

    def training_step(self, batch, batch_idx, optimizer_idx):
        batch = self.convert_to_haar_space(batch, max_depth=self.num_scales-1)

        scale_index = batch_idx % self.num_scales + 1
        self.get_relevant_scales_from_batch(batch, scale_index)
        T1, T2 = self.compute_interval(scale_index)
        scale_name = self.index_to_scale_name[scale_index]
        loss = self.train_loss_fn(self.score_model[scale_name], batch, T1, T2)
        self.log('train_loss_scale_%s' % scale_name, loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = self.convert_to_haar_space(batch, max_depth=self.num_scales-1)

        scale_index = np.random.randint(low=1, high=self.num_scales+1)
        self.get_relevant_scales_from_batch(batch, scale_index)
        T1, T2 = self.compute_interval(scale_index)
        scale_name = self.index_to_scale_name[scale_index]
        loss = self.eval_loss_fn(self.score_model[scale_name], batch, T1, T2)
        self.log('eval_loss_scale_%s' % scale_name, loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        return 
    
    def get_relevant_scales_from_batch(self, batch, scale_index):
        for i in range(1, self.num_scales+1):
            if i < scale_index:
                key = self.index_to_scale_name[i]
                del batch[key]


    def haar_forward(self, x):
        haar_transform = InvertibleDownsampling2D(3, stride=2, method='cayley', init='haar', learnable=False).to(self.device)
        x = haar_transform(x)
        x = self.permute_channels(x)
        return x

    def permute_channels(self, haar_image, forward=True):
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

    def convert_to_haar_space(self, x, max_depth):
        haar_x = {}
        for i in range(max_depth):
            x = self.haar_forward(x)
            if i < max_depth - 1:
                haar_x['d%d'%(i+1)] = x[:,3:,::]
                x = x[:,:3,::]
            elif i == max_depth - 1:
                haar_x['d%d'%(i+1)] = x[:,3:,::]
                haar_x['a%d'%(i+1)] = x[:,:3,::]
        
        return haar_x

    def sample(self, num_samples=None, predictor='default', 
                corrector='default', p_steps='default', 
                c_steps='default', probability_flow='default',
                snr='default', show_evolution=False, 
                denoise='default', adaptive='default', 
                gamma=0., alpha=1., 
                starting_T='default', ending_T='default'):
        
        scale_sampling_order = self.reverse_scale_order
        x=None
        aggregate_sampling_information = {}
        for new_scale_name in scale_sampling_order:
            scale_index = self.scale_name_to_index[new_scale_name]
            T1, T2 = self.compute_interval(scale_index) #T1 < T2 -> T2:start, T1:end
            config = self.config[new_scale_name]

            start_time = timer()
            x, sampling_info = self.scale_sample(x=x, num_samples=num_samples, predictor=predictor, 
                                corrector=corrector, p_steps=p_steps, 
                                c_steps=c_steps, probability_flow=probability_flow,
                                snr=snr, show_evolution=show_evolution, 
                                denoise=denoise, adaptive=adaptive, 
                                gamma=gamma, alpha=alpha, 
                                starting_T=T2, ending_T=T1, new_scale_name=new_scale_name, config=config)
            end_time = timer()

            aggregate_sampling_information[new_scale_name] = {}
            aggregate_sampling_information[new_scale_name]['sampling_information'] = sampling_info
            aggregate_sampling_information[new_scale_name]['time'] = end_time - start_time
        
        x = self.score_model['d1'].convert_to_image_space(x)
        return x, aggregate_sampling_information


    def scale_sample(self, x=None, num_samples=None, predictor='default', 
                     corrector='default', p_steps='default', 
                     c_steps='default', probability_flow='default',
                     snr='default', show_evolution=False, 
                     denoise='default', adaptive='default', 
                     gamma=0., alpha=1., 
                     starting_T='default', ending_T='default', new_scale_name=None, config=None):
        
        if num_samples is None:
            num_samples = config.eval.batch_size

        if starting_T == 'default':
            starting_T = self.sde[new_scale_name].T

        #Code for managing adaptive sampling
        if adaptive == 'default':
            if hasattr(config.sampling, 'adaptive'):
                adaptive = config.sampling.adaptive
                assert adaptive in [True, False], 'adaptive flag should be either True or False'
            else:
                adaptive = False
            
        if adaptive:
            if config.eval.adaptive_method == 'kl':
                if config.sampling.kl_profile == None:
                    adaptive = False
            elif config.eval.adaptive_method == 'lipschitz':
                if config.sampling.lipschitz_profile == None:
                    adaptive = False

            if adaptive:
                print('Adaptive discretisation is used.')
                try:
                    adaptive_discretisation_fn = self.adaptive_dicrete_fn

                    if config.eval.adaptive_method == 'kl':
                        if gamma != self.gamma:
                            #gamma has changed so we need to update the adaptive discretisation function
                            adaptive_discretisation_fn = get_adaptive_discretisation_fn(self.kl_info['t'], self.kl_info['KL'], gamma, 'kl')
                            self.adaptive_dicrete_fn = adaptive_discretisation_fn
                            self.gamma = gamma
                    
                    if config.eval.adaptive_method == 'lipschitz':
                        if alpha != self.alpha:
                            #alpha has changed so we need to update the adaptive discretisation function
                            adaptive_discretisation_fn = get_adaptive_discretisation_fn(self.lipschitz_info['t'], self.lipschitz_info['Lip_constant'], alpha, 'lipschitz')
                            self.adaptive_dicrete_fn = adaptive_discretisation_fn
                            self.alpha = alpha
                            
                except AttributeError:

                    if config.eval.adaptive_method == 'kl':
                        #load the KL profile
                        with open(config.sampling.kl_profile, 'rb') as f:
                            info = pickle.load(f)

                        self.kl_info = info
                        adaptive_discretisation_fn = get_adaptive_discretisation_fn(info['t'], info['KL'], gamma, 'kl')
                        self.adaptive_dicrete_fn = adaptive_discretisation_fn
                        self.gamma = gamma
                    
                    elif config.eval.adaptive_method == 'lipschitz':
                        #load the lipschitz profile
                        with open(config.sampling.lipschitz_profile, 'rb') as f:
                            info = pickle.load(f)
                        
                        self.lipschitz_info = info
                        adaptive_discretisation_fn = get_adaptive_discretisation_fn(self.lipschitz_info['t'], self.lipschitz_info['Lip_constant'], alpha, 'lipschitz')
                        self.adaptive_dicrete_fn = adaptive_discretisation_fn
                        self.alpha = alpha

            else:
                print('uniform-discretisation is used.')
                adaptive_discretisation_fn=None
        else:
            print('uniform-discretisation is used.')
            adaptive_discretisation_fn=None 
        
        if adaptive:
            if config.eval.adaptive_method == 'kl':
                adaptive_steps = torch.tensor(self.adaptive_dicrete_fn(p_steps+1))

            elif config.eval.adaptive_method == 'lipschitz':
                T_start = self.sde[new_scale_name].T if starting_T == 'default' else starting_T
                adaptive_steps = torch.tensor(self.adaptive_dicrete_fn(T_start))
        else:
            adaptive_steps = None

        sampling_shape = [num_samples] + config.data.scale_shape
        sampling_fn = get_sampling_fn(config=config, 
                                      sde=self.sde, 
                                      shape=sampling_shape, 
                                      eps=self.sampling_eps,
                                      predictor=predictor, 
                                      corrector=corrector, 
                                      p_steps=p_steps, 
                                      c_steps=c_steps, 
                                      probability_flow=probability_flow,
                                      snr=snr,
                                      show_evolution=show_evolution,
                                      denoise=denoise, 
                                      adaptive_steps=adaptive_steps,
                                      starting_T = starting_T,
                                      ending_T = ending_T,
                                      multiscale=True)

        return sampling_fn(self.score_model[new_scale_name], x, new_scale_name)

    def configure_optimizers(self):
        class scheduler_lambda_function:
            def __init__(self, warm_up):
                self.use_warm_up = True if warm_up > 0 else False
                self.warm_up = warm_up

            def __call__(self, s):
                if self.use_warm_up:
                    if s < self.warm_up:
                        return s / self.warm_up
                    else:
                        return 1
                else:
                    return 1
        
        optimisation_info = []
        for i in range(1, self.num_scales+1):
            scale_name = self.index_to_scale_name[i]
            optimizer = losses.get_optimizer(self.config[scale_name], self.score_model[scale_name].parameters())
            scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(optimizer, scheduler_lambda_function(self.config[scale_name].optim.warmup)),
                        'interval': 'step'}  # called after each training step
            
            optimisation_info.append({"optimizer": optimizer, "lr_scheduler": scheduler, "frequency":1})

        return optimisation_info

    
    

