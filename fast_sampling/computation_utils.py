import torch
import sde_lib
import numpy as np
from models import utils as mutils
from pathlib import Path
import pickle 
import os
from lightning_data_modules.utils import create_lightning_datamodule
from lightning_modules.utils import create_lightning_module
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_expectations(timestamps, model, sde, dataloader, mu_0, device):
    expectations = {}
    for timestamp in tqdm(timestamps):
        dict_timestamp = timestamp.item()
        expectations[dict_timestamp]={}
        results = compute_sliced_expectations(timestamp, model, sde, dataloader, mu_0, device)
        expectations[dict_timestamp]['x_2'] = results['x_2']
        expectations[dict_timestamp]['score_x_2'] = results['score_x_2']
    return expectations

def compute_sliced_expectations(timestamp, model, sde, dataloader, mu_0, device):
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)

    num_datapoints = 0
    exp_x_2 = 0.
    exp_norm_grad_log_density = 0.
    dims=None
    for idx, batch in tqdm(enumerate(dataloader)):
        if idx>1:
            break

        if dims is None:
            dims = np.prod(batch.shape[1:])

        t = timestamp.repeat(batch.size(0)) 
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        x = mean + std[(...,) + (None,) * len(batch.shape[1:])] * z

        num_datapoints += x.size(0)

        if mu_0 is not None:
            exp_x_2 += torch.sum(torch.square(x-mu_0))
        else:
            exp_x_2 += torch.sum(torch.square(x))

        with torch.no_grad():
            score_x = score_fn(x.to(device), t.to(device))

        exp_norm_grad_log_density += torch.sum(torch.square(score_x.to('cpu')))

    exp_x_2 /= num_datapoints
    exp_norm_grad_log_density /= num_datapoints

    return {'x_2': exp_x_2, 'score_x_2': exp_norm_grad_log_density, 't': timestamp}

def find_timestamps_geq(t, timestamps):
    #return timestamps greater or equal to time t (we assumed that timestamps are ordered increasingly)
    for i, timestamp in enumerate(timestamps):
        if timestamp >= t:
            break
    return timestamps[i:]

def get_KL_divergence_fn(model, dataloader, shape, sde, eps, 
                         discretisation_steps, save_dir, device, 
                         load_expections=False, mu_0=None):

    ##get the function that approximates the KL divergence from time t to time T, i.e., KL(p_t||p_T)
    
    def get_integrant_discrete_values(timestamps, sde, expectations):
        integrant_discrete_values = {}
        for timestamp in timestamps:
            _, g = sde.sde(torch.zeros(1), timestamp)
            integrant_discrete_values[timestamp] = g**2 * expectations[timestamp]['score_x_2']
        return integrant_discrete_values
    
    def get_integral_calculator(integrant_discrete_values, timestamps):
        def integral(t, T):
            if t == T:
                return 0.
            else:
                integrated_timestamps = find_timestamps_geq(t, timestamps)
                M = len(integrated_timestamps)
                int_sum = torch.sum([integrant_discrete_values[t] for t in integrated_timestamps])
                return (T-t)/M * int_sum

        return integral

    timestamps = torch.linspace(start=eps, end=sde.T, steps=discretisation_steps)
    
    if load_expections:
        #load them from the saved directory
        with open(os.path.join(save_dir, 'expectations.pkl'), 'rb') as f:
            expectations = pickle.load(f)
    else:
        expectations = compute_expectations(timestamps, model, sde, dataloader, mu_0, device)

        #save the expectations. It is computationally expensive to re-compute them.
        with open(os.path.join(save_dir, 'expectations.pkl'), 'wb') as f:
            pickle.dump(expectations, f)
    
    timestamps = timestamps.numpy()
    dims = np.prod(shape)
    integrant_discrete_values = get_integrant_discrete_values(timestamps, sde, expectations)
    integral_fn = get_integral_calculator(integrant_discrete_values, timestamps)

    def KL(t):
        assert t in timestamps, 't is not in timestamps. Interpolation is not supported yet for x_2 expectation.'
        if isinstance(sde, sde_lib.VESDE):
            sigma_t = sde.marginal_prob(torch.zeros(1), t)
            sigma_T = sde.marginal_prob(torch.zeros(1), sde.T)

            A = dims/2*torch.log(2*np.pi*sigma_t**2)+1/2*sigma_t**(-2)*expectations[t]['x_2']
            A -= dims/2*torch.log(2*np.pi*sigma_T**2)+dims/2

        A += 1/2*integral_fn(t, sde.T) #this is common for both VE and VP sdes. The other terms are incorporated in the previous if statement.

        return A
        
    return KL

def calculate_mean(dataloader):
    mean=None
    total_num_images=0
    for batch in dataloader:
        if mean is None:
            mean = torch.zeros_like(batch[0])

        mean += torch.sum(batch, dim=0)
        num_images = batch.size(0)
        total_num_images +=num_images
    
    mean /= total_num_images
    return mean

def fast_sampling_scheme(config, save_dir):
    device = 'cpu'
    dsteps = 10
    use_mu_0 = True

    if config.base_log_path is not None:
        save_dir = os.path.join(config.base_log_path, config.experiment_name, 'KL')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    assert config.model.checkpoint_path is not None, 'checkpoint path has not been provided in the configuration file.'
    
    DataModule = create_lightning_datamodule(config)
    DataModule.setup()
    dataloader = DataModule.val_dataloader()

    lmodule = create_lightning_module(config, config.model.checkpoint_path).to(device)
    lmodule.eval()
    lmodule.configure_sde(config)

    model = lmodule.score_model
    sde = lmodule.sde
    eps = lmodule.sampling_eps

    if use_mu_0 and isinstance(sde, sde_lib.VESDE):
        mu_0 = calculate_mean(dataloader) #this will be used for the VE SDE if use_mu_0 flag is set to True.
    else:
        mu_0 = None

    KL = get_KL_divergence_fn(model=model, 
                              dataloader=dataloader,
                              shape=config.data.shape, 
                              sde=sde,
                              eps=eps,
                              discretisation_steps=dsteps,
                              save_dir = save_dir,
                              device=device,
                              load_expections=False,
                              mu_0=mu_0)
    
    timestamps = torch.linspace(start=eps, end=sde.T, steps=dsteps)
    KLs = [KL(t) for t in timestamps]

    plt.figure()
    plt.plot(timestamps, KLs)
    plt.xlabel('diffusion time')
    plt.ylabel('KL divergence')
    plt.savefig(os.path.join(save_dir, 'KL_vs_t.png'))







    




