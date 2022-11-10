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
from utils import compute_grad

""" FUNCTIONS USED FOR EVALUATING THE KL DIVERGENCE """

def normalise_to_density(t, x):
    a, b, M = t.min(), t.max(), len(x)
    integral = (b-a)/M*np.sum(x)
    return x/integral

def get_uniformisation_fn(gamma):
    def uniformisation_fn(fn):
        mean=np.mean(fn)
        return gamma*fn+(1-gamma)*mean
    return uniformisation_fn

def evaluate_continuation(x_2, norm_grad_log_density, num_datapoints):
    def evaluate_quantity_continuation(quantity, num_datapoints, r):
        #we want to calculate the number of samples needed for the confidence interval of the sample mean to have width W = r * mean
        mean = torch.mean(quantity)
        std = torch.std(quantity)

        optimal_num_samples = int(4*1.96**2*std**2/(r*mean)**2)

        if optimal_num_samples > num_datapoints:
            return optimal_num_samples
        else:
            return 0

    r = 0.05
    x_2 = torch.cat(x_2, dim=0)  
    norm_grad_log_density = torch.cat(norm_grad_log_density, dim=0)

    cont_x_2 = evaluate_quantity_continuation(x_2, num_datapoints, r)
    cont_score_norm = evaluate_quantity_continuation(norm_grad_log_density, num_datapoints, r)

    if cont_x_2 + cont_score_norm > 0:
        return True
    elif cont_x_2 + cont_score_norm == 0:
        return False
    else:
        return 'Something is going wrong.'

def compute_expectations(timestamps, model, sde, dataloader, mu_0, device):
    expectations = {}
    for timestamp in tqdm(timestamps):
        dict_timestamp = timestamp.item()
        expectations[dict_timestamp]={}
        results = compute_sliced_expectations(timestamp, model, sde, dataloader, mu_0, device)
        expectations[dict_timestamp]['x_2'] = results['x_2']
        expectations[dict_timestamp]['score_x_2'] = results['score_x_2']

        #addition
        expectations[dict_timestamp]['exp_stats'] = results['exp_stats']
    
    #print(expectations)
    return expectations

def get_curvature_profile(config):
    assert config.model.checkpoint_path is not None, 'checkpoint path has not been provided in the configuration file.'

    device = 'cuda'
    DataModule = create_lightning_datamodule(config)
    DataModule.setup()
    dataloader = DataModule.train_dataloader()

    lmodule = create_lightning_module(config, config.model.checkpoint_path).to(device)
    lmodule.eval()
    lmodule.configure_sde(config)

    model = lmodule.score_model
    sde = lmodule.sde
    eps = lmodule.sampling_eps
    t_grid = 20
    num_batches = 125

    if config.base_log_path is not None:
        save_dir = os.path.join(config.base_log_path, config.experiment_name, 'curvature')

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    curvature_estimator = get_curvature_profile_fn(dataloader, model, sde, num_batches, True, device)
    timesteps = torch.linspace(eps, sde.T, t_grid, device=device)

    curvatures = []
    for i in tqdm(range(timesteps.size(0))):
        t = timesteps[i]
        curvatures.append(curvature_estimator(t))
    
    with open(os.path.join(save_dir, 'info.pkl'), 'wb') as f:
        info = {'t':timesteps.cpu().tolist(), 'curvatures':curvatures}
        pickle.dump(info, f)


def get_curvature_profile_fn(dataloader, model, sde, num_batches, continuous=True, device='cuda'):
    #output: a fn that receives model as input and outputs the estimated curvature profile

    def get_f_ode_fn(sde, model):
        score_fn = mutils.get_score_fn(sde, model, conditional=False, train=False, continuous=continuous)
        def f_ode_fn(x, t):
            score = score_fn(x, t)
            drift, diffusion = sde.sde(x, t)
            return drift - diffusion[:, None, None, None] ** 2 * score * 0.5
        return f_ode_fn
    
    def get_fode_as_fn_of_x(f_ode, t):
        def fode_as_fn_of_x(x):
            return f_ode(x, t)
        return fode_as_fn_of_x

    def get_fode_as_fn_of_t(f_ode, x):
        def fode_as_fn_of_t(t):
            return f_ode(x, t)
        return fode_as_fn_of_t

    def project_acc_vector(velocity, acceleration):
        #velocity and acceleration are assumed batched tensors.
        #velocity.size() = (batch, velocity)
        #acceleration.size() = (batch, acceleration)

        normalised_velocity = torch.nn.functional.normalize(velocity, p=2.0, dim=1)
        tangential_acceleration_coefficient = torch.sum(normalised_velocity*acceleration, dim=1)
        tangential_acceleration = tangential_acceleration_coefficient[:,None]*normalised_velocity
        centripetal_acceleration = acceleration - tangential_acceleration
        return tangential_acceleration, centripetal_acceleration

    def centripetal_acceleration_fn(x, t):
        f_ode_fn = get_f_ode_fn(sde, model)
        f_ode_x = get_fode_as_fn_of_x(f_ode_fn, t)
        f_ode_t = get_fode_as_fn_of_t(f_ode_fn, x)

        velocity = f_ode_fn(x, t)
        jvp = torch.autograd.functional.jvp(f_ode_x, x, v=velocity)[1].detach()
        df_dt = torch.autograd.functional.jvp(f_ode_t, t, v=torch.ones_like(t))[1].detach()
        
        acceleration = jvp + df_dt
        centripetal_acceleration = project_acc_vector(torch.flatten(velocity, start_dim=1), torch.flatten(acceleration, start_dim=1))[1]
        return centripetal_acceleration
    
    def average_curvature(x, t): #t is the same for all x
        centripetal_acceleration = centripetal_acceleration_fn(x, t)
        centr_acc_magnitudes = torch.linalg.norm(centripetal_acceleration, dim=1)
        return torch.mean(centr_acc_magnitudes)

    def curvature_estimator_fn(t):
        t_curvatures = []
        for idx, batch in enumerate(dataloader):
            if idx > num_batches:
                break

            batch = batch.to(device)
            z = torch.randn_like(batch.to(device))
            vec_t = torch.ones(batch.size(0), device=model.device) * t
            mean, std = sde.marginal_prob(batch, vec_t)
            x = mean + std[(...,) + (None,) * len(batch.shape[1:])] * z

            avg_curvature = average_curvature(x, vec_t).item()
            t_curvatures.append(avg_curvature)
      
        return torch.mean(torch.tensor(t_curvatures)).cpu().item()
    
    return curvature_estimator_fn

def get_Lip_constant_profile(config):
    assert config.model.checkpoint_path is not None, 'checkpoint path has not been provided in the configuration file.'

    device = 'cuda'
    dsteps = 1000
    
    DataModule = create_lightning_datamodule(config)
    DataModule.setup()
    dataloader = DataModule.train_dataloader()

    lmodule = create_lightning_module(config, config.model.checkpoint_path).to(device)
    lmodule.eval()
    lmodule.configure_sde(config)

    model = lmodule.score_model
    sde = lmodule.sde
    eps = lmodule.sampling_eps

    if config.base_log_path is not None:
        save_dir = os.path.join(config.base_log_path, config.experiment_name, 'Lip_constant')
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    Lip_constant_fn = get_Lip_constant_fn(model=model, 
                              dataloader=dataloader, 
                              sde=sde)
    
    timestamps = torch.linspace(start=eps, end=sde.T, steps=dsteps)
    
    Lip_constant_profile = []
    for i in tqdm(range(timestamps.size(0))):
        t = timestamps[i]
        Lip_constant_profile.append(Lip_constant_fn(t))

    with open(os.path.join(save_dir, 'info.pkl'), 'wb') as f:
        info = {'t':timestamps, 'Lip_constant':Lip_constant_profile}
        pickle.dump(info, f)


def get_Lip_constant_fn(model, dataloader, sde):
    def get_drift_fn(model):
        """Get the drift function of the reverse-time SDE."""
        score_fn = mutils.get_score_fn(sde, model, conditional=False, train=False, continuous=True)
        def drift_fn(x, t):
            rsde = sde.reverse(score_fn, probability_flow=True)
            return rsde.sde(x, t)[0]
        return drift_fn
    
    def Lip_constant_fn(t):
        drift_fn = get_drift_fn(model)
        
        max_grad_norm = 0.
        for idx, batch in enumerate(dataloader):
            if idx > 400:
                break

            batch = batch.to(model.device)
            z = torch.randn_like(batch.to(model.device))
            vec_t = torch.ones(batch.size(0), device=model.device) * t
            mean, std = sde.marginal_prob(batch, vec_t)
            x = mean + std[(...,) + (None,) * len(batch.shape[1:])] * z

            gradients = compute_grad(f=drift_fn, x=x, t=vec_t)
            gradients = gradients.reshape(gradients.shape[0], -1)
            grad_norm = gradients.norm(p=2, dim=1).max().item() #this should probably be modified
            
            if grad_norm > max_grad_norm:
                max_grad_norm = grad_norm

        return max_grad_norm

    return Lip_constant_fn

def compute_sliced_expectations(timestamp, model, sde, dataloader, mu_0, device):
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=True)

    num_datapoints = 0
    exp_x_2 = 0.
    exp_norm_grad_log_density = 0.
    dims=None

    #those settings will be used for adapting the stepsize
    check_point = 500
    x_2=[]
    norm_grad_log_density=[]
    for idx, batch in enumerate(dataloader):
        if num_datapoints >= check_point:
            continuation = evaluate_continuation(x_2, norm_grad_log_density, num_datapoints)
            if continuation == False:
                print('Adaptive mean evaluation stops at %d samples - the 95 confidence interval is [mean-r/2*mean, mean+r/2mean]' % num_datapoints)
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

            #addition
            x_2.append(torch.sum(torch.square(x-mu_0), dim=[i+1 for i in range(len(x.shape)-1)]))
        else:
            exp_x_2 += torch.sum(torch.square(x))

            #addition
            x_2.append(torch.sum(torch.square(x), dim=[i+1 for i in range(len(x.shape)-1)]))

        with torch.no_grad():
            score_x = score_fn(x.to(device), t.to(device))
            score_x = score_x.to('cpu')

        exp_norm_grad_log_density += torch.sum(torch.square(score_x))
        
        #addition
        norm_grad_log_density.append(torch.sum(torch.square(score_x), dim=[i+1 for i in range(len(score_x.shape)-1)]))

    exp_x_2 /= num_datapoints
    exp_norm_grad_log_density /= num_datapoints

    #addition
    x_2 = torch.cat(x_2, dim=0)  
    norm_grad_log_density = torch.cat(norm_grad_log_density, dim=0)

    exp_stats = {'num_datapoints': num_datapoints,
                 'x_2': {'mean': torch.mean(x_2), 
                         'std': torch.std(x_2, unbiased=True)},

                 'score_norm': {'mean': torch.mean(norm_grad_log_density), 
                                'std': torch.std(norm_grad_log_density, unbiased=True)}
                }

    #addition
    return {'x_2': exp_x_2.item(), 'score_x_2': exp_norm_grad_log_density.item(), 't': timestamp, 'exp_stats':exp_stats}

def find_timestamps_geq(t, timestamps):
    #return timestamps greater or equal to time t (we assumed that timestamps are ordered increasingly)
    for i, timestamp in enumerate(timestamps):
        if timestamp >= t:
            break
    return timestamps[i:]



def get_KL_divergence_fn(model, dataloader, shape, sde, eps, T,
                         discretisation_steps, save_dir, device, 
                         load_expections=False, mu_0=None, target_dist='T'):

    ##get the function that approximates the KL divergence from time t to time T, i.e., KL(p_t||p_T)
    
    def get_integrant_discrete_values(timestamps, sde, expectations):
        integrant_discrete_values = {}
        for timestamp in timestamps:
            _, g = sde.sde(torch.zeros(1), torch.tensor(timestamp, dtype=torch.float32))
            integrant_discrete_values[timestamp] = (g**2).item() * expectations[timestamp]['score_x_2']
        return integrant_discrete_values
    
    def get_integral_calculator(integrant_discrete_values, timestamps):
        def integral(t, T):
            if t == T:
                return 0.
            else:
                integrated_timestamps = find_timestamps_geq(t, timestamps)
                M = len(integrated_timestamps)
                int_sum = np.sum([integrant_discrete_values[tau] for tau in integrated_timestamps])
                return (T-t)/M * int_sum

        return integral

    timestamps = torch.linspace(start=eps, end=T, steps=discretisation_steps)
    
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
            if target_dist == 'T':
                target_distribution_t = T
            elif target_dist == 't':
                target_distribution_t = t

            _, sigma_t = sde.marginal_prob(torch.zeros(1), torch.tensor(target_distribution_t, dtype=torch.float32))
            _, sigma_T = sde.marginal_prob(torch.zeros(1), torch.tensor(T, dtype=torch.float32))

            sigma_t, sigma_T = sigma_t.item(), sigma_T.item()
            A = dims/2*np.log(2*np.pi*sigma_t**2)+1/2*sigma_t**(-2)*expectations[t]['x_2']
            A -= dims/2*np.log(2*np.pi*sigma_T**2)+dims/2

        elif isinstance(sde, sde_lib.VPSDE):
            A = -dims/2 + 1/2*expectations[t]['x_2'] 
            #the next term comes from the integral
            A -= 1/4*sde.beta_0*dims*(sde.T-t) + 1/8*(sde.beta_1-sde.beta_0)*dims*(sde.T**2-t**2)

        A += 1/2*integral_fn(t, T) #this is common for both VE and VP sdes. The other terms are incorporated in the previous if statement.

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

""" FUNCTIONS USED FOR EVALUATING THE ADAPTIVE SAMPLING SCHEME FROM THE KL DIVERGENCE """

def find_inter_tstamps(t1, t0, timestamps):
    intermediate_tstamps = []
    idx_intermediate_tstamps = []
    for i, tstamp in enumerate(timestamps):
        if tstamp>=t0 and tstamp<=t1:
            intermediate_tstamps.append(tstamp)
            idx_intermediate_tstamps.append(i)

    return intermediate_tstamps, idx_intermediate_tstamps

def find_closest_cdf_values(x, cdf):
    #assert cdf.min()<x and cdf.max()>x, 'x is not in correct range (%.7f,%.7f): x=%.7f' % (cdf[0],cdf[-1],x)
    if x < cdf.min():
        return cdf[0], cdf[0]
    else:
        stop = False
        len_cdf = len(cdf)
        for i in range(len_cdf):
            if not stop:
                if cdf[len_cdf - i - 1] < x:
                    #print(cdf[len_cdf - i - 1])
                    next_val = cdf[len_cdf-i]
                    prev_val = cdf[len_cdf-i-1]
                    stop=True
            else:
                break

        return prev_val, next_val

def map_cdf_to_time(timestamps, cdf):
    cdf_to_time = {}
    for i, t in enumerate(timestamps):
        cdf_to_time[cdf[i]]=timestamps[i]
    return cdf_to_time

def get_inverse_cdf_fn(timestamps, cdf):
    cdf_to_time = map_cdf_to_time(timestamps, cdf)
    def inverse_cdf_fn(x):
        if x in cdf:
            return cdf_to_time[x]
        else:
            prev_val, next_val = find_closest_cdf_values(x, cdf)
            if prev_val == next_val and prev_val == cdf[0]:
                return cdf_to_time[prev_val]
            else:
                prev_distance, next_distance = np.abs(x-prev_val), np.abs(x-next_val)
                total_dist = prev_distance + next_distance
                prev_frac_dist, next_frac_dist = prev_distance/total_dist, next_distance/total_dist

                t0, t1 = cdf_to_time[prev_val], cdf_to_time[next_val]
                return t0+prev_frac_dist*(t1-t0)
    
    return inverse_cdf_fn

def get_cdf_fn(timestamps, density):
    eps, T = timestamps[0], timestamps[-1]
    def cdf_fn(t):
        assert t >= eps and t <= T, 't is not in the range (eps,T)'
        inter_tstamps, idx_inter_tstamps = find_inter_tstamps(t, eps, timestamps)
        inter_densities = [density[x] for x in idx_inter_tstamps]
        M = len(inter_densities)
        if M==0:
            return 0.
        else:
            return (t-eps)/M*np.sum(inter_densities)
    return cdf_fn

def get_adaptive_step_calculator_from_density(timestamps, density):
    eps, T = timestamps[0], timestamps[-1]
    S = (T-eps) / len(density) * np.sum(density) #integral S
    print('S: %.4f' % S)
    
    cdf_fn = get_cdf_fn(timestamps, density)
    cdf = np.array([cdf_fn(x) for x in timestamps])
    #print(cdf[0], cdf[-1])
    inverse_cdf_fn = get_inverse_cdf_fn(timestamps, cdf)

    def calculate_adaptive_steps(N):
        intervals = N-1
        stepsize = cdf[-1]/intervals
        print('stepsize: %.4f' % stepsize)
        adapt_steps=[T]
        cumult = 1
        for i in tqdm(range(intervals)):
            cumult -= stepsize
            #print(cumult)
            t_cumult = inverse_cdf_fn(cumult)
            adapt_steps.append(t_cumult)

        return adapt_steps
        
    return calculate_adaptive_steps

def get_adaptive_discretisation_fn(timestamps, value, gamma, adaptive_method):
    if adaptive_method == 'kl':
        KL = value
        #input: KL divergence at timestamps (usually calculated from 0 to 1 for 1000 equally spaced points)
        #outputs: adaptive discretisation function (receives as input the number of discretisation points and outputs their time locations)
        grad_KL = np.gradient(KL)
        abs_grad_KL = np.abs(grad_KL)
        uniformisation_fn = get_uniformisation_fn(gamma=gamma)
        normalised_abs_grad_KL = normalise_to_density(timestamps, uniformisation_fn(abs_grad_KL))
        return get_adaptive_step_calculator_from_density(timestamps, normalised_abs_grad_KL)
    
    elif adaptive_method == 'lipschitz':
        lipschitz_constant = value
        alpha = gamma
        def sequence_generator(starting_T):
            start = starting_T
            end = timestamps[0]

            sequence = []

            current_timepoint = start
            sequence.append(current_timepoint)
            while current_timepoint > end:
                lip = lipschitz_constant[np.argmin(np.abs(timestamps-current_timepoint))]
                step = -1 * alpha * 1/lip

                current_timepoint = current_timepoint + step

                if current_timepoint >= end:
                    sequence.append(current_timepoint)
                else:
                    sequence.append(end)
            
            return sequence
        
        return sequence_generator


def get_inverse_step_fn(discretisation):
    #discretisation sequence is ordered from biggest time to smallest time
    map_t_to_negative_dt = {}
    steps = len(discretisation)
    for i in range(steps):
        if i <= steps-2:
            map_t_to_negative_dt[discretisation[i]] = discretisation[i+1] - discretisation[i]
        elif i==steps-1:
            map_t_to_negative_dt[discretisation[i]] = map_t_to_negative_dt[discretisation[i-1]]

    def inverse_step_fn(t):
        if t in map_t_to_negative_dt.keys():
            return map_t_to_negative_dt[t]
        else:
            closest_t_key = discretisation[np.argmin(np.abs(discretisation-t))]
            #print('closest_t_key: ', closest_t_key)
            #print('t: ', t)
            return map_t_to_negative_dt[closest_t_key]
    
    return inverse_step_fn


def fast_sampling_scheme(config, save_dir):
    device = 'cuda'
    dsteps = 1000
    use_mu_0 = True
    target_distribution = 'T'
    T = 'sde' #0.675 for the ve sde

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

    if T=='sde':
        T = sde.T

    if use_mu_0 and isinstance(sde, sde_lib.VESDE):
        mu_0 = calculate_mean(dataloader) #this will be used for the VE SDE if use_mu_0 flag is set to True.
    else:
        mu_0 = None

    if config.base_log_path is not None:
        save_dir = os.path.join(config.base_log_path, config.experiment_name, 'KL','T=%.3f-Target_Distribution=%s' % (T,target_distribution))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    KL = get_KL_divergence_fn(model=model, 
                              dataloader=dataloader,
                              shape=config.data.shape, 
                              sde=sde,
                              eps=eps,
                              T=T,
                              discretisation_steps=dsteps,
                              save_dir = save_dir,
                              device=device,
                              load_expections=False,
                              mu_0=mu_0,
                              target_dist=target_distribution)
    
    timestamps = torch.linspace(start=eps, end=T, steps=dsteps).numpy()
    KL = [KL(t) for t in timestamps]
    grad_KL = np.gradient(KL)
    abs_grad_KL = np.abs(grad_KL)
    normalised_abs_grad_KL = normalise_to_density(timestamps, abs_grad_KL)

    with open(os.path.join(save_dir, 'info.pkl'), 'wb') as f:
        info = {'t':timestamps, 'KL':KL, 'grad_KL':grad_KL, \
                'abs_grad_KL':abs_grad_KL, 'normalised_abs_grad_KL':normalised_abs_grad_KL}
        pickle.dump(info, f)

    plt.figure()
    plt.plot(timestamps, KL)
    plt.xlabel('diffusion time')
    plt.ylabel('KL divergence')
    plt.savefig(os.path.join(save_dir, 'KL_vs_t.png'))

    plt.figure()
    plt.plot(timestamps, grad_KL)
    plt.xlabel('diffusion time')
    plt.ylabel('grad KL divergence')
    plt.savefig(os.path.join(save_dir, 'grad_KL_vs_t.png'))
    
    plt.figure()
    plt.plot(timestamps, abs_grad_KL)
    plt.xlabel('diffusion time')
    plt.ylabel('abs grad KL divergence')
    plt.savefig(os.path.join(save_dir, 'abs_grad_KL_vs_t.png'))

    plt.figure()
    plt.plot(timestamps, normalised_abs_grad_KL)
    plt.xlabel('diffusion time')
    plt.ylabel('norm abs grad KL divergence')
    plt.savefig(os.path.join(save_dir, 'norm_abs_grad_KL_vs_t.png'))









    




