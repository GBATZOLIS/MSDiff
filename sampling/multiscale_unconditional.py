from sampling.predictors import get_predictor, ReverseDiffusionPredictor, NonePredictor
from sampling.correctors import get_corrector, NoneCorrector
from tqdm import tqdm
import functools
import torch
import numpy as np
import abc
from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
import pickle 

def get_sampling_fn(config, sde, shape, eps,
                    predictor='default', 
                    corrector='default', 
                    p_steps='default', 
                    c_steps='default', 
                    probability_flow='default',
                    snr='default', 
                    denoise='default', 
                    adaptive_steps=None,
                    starting_T='default',
                    ending_T='default',
                    scale=1):

  """Create a sampling function.
  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.
  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

  if predictor == 'default':
    predictor = get_predictor(config.sampling.predictor.lower())
  else:
    predictor = get_predictor(predictor.lower())

  if corrector == 'default':
    corrector = get_corrector(config.sampling.corrector.lower())
  else:
    corrector = get_corrector(corrector.lower())

  if p_steps == 'default':
    p_steps = config.model.num_scales

  if c_steps == 'default':
    c_steps = config.sampling.n_steps_each

  if snr == 'default':
    snr = config.sampling.snr

  if denoise == 'default':
    denoise = config.sampling.noise_removal

  if probability_flow == 'default':
    probability_flow=config.sampling.probability_flow

  sampler_name = config.sampling.method

  # Probability flow ODE sampling with black-box ODE solvers
  if sampler_name.lower() == 'ode':
    sampling_fn = get_ode_sampler(sde=sde,
                                  shape=shape,
                                  denoise=denoise,
                                  eps=eps)

  # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
  elif sampler_name.lower() == 'pc':
    sampling_fn = get_pc_sampler(sde=sde,
                                 shape=shape,
                                 predictor=predictor,
                                 corrector=corrector,
                                 snr=snr,
                                 p_steps=p_steps,
                                 c_steps=c_steps, 
                                 probability_flow=probability_flow,
                                 continuous=config.training.continuous,
                                 denoise=denoise,
                                 eps=eps,
                                 adaptive_steps=adaptive_steps,
                                 starting_T=starting_T,
                                 ending_T=ending_T,
                                 scale=scale)
  else:
    raise ValueError(f"Sampler name {sampler_name} unknown.")

  return sampling_fn


def get_ode_sampler(sde, shape,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3):
  """Probability flow ODE sampler with the black-box ODE solver.
  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, conditional=False, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def drift_fn(model, x, t):
    """Get the drift function of the reverse-time SDE."""
    score_fn = get_score_fn(sde, model, conditional=False, train=False, continuous=True)
    rsde = sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def ode_sampler(model, z=None):
    """The probability flow ODE sampler with black-box ODE solver.
    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      if z is None:
        # If not represent, sample the latent code from the prior distibution of the SDE.
        x = sde.prior_sampling(shape).to(model.device)
      else:
        x = z

      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(model.device).type(torch.float32)
        vec_t = torch.ones(shape[0], device=x.device) * t
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(model.device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      if denoise:
        x = denoise_update_fn(model, x)

      return x, nfe

  return ode_sampler


def get_pc_sampler(sde, shape, predictor, corrector, snr, 
                   p_steps, c_steps, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, adaptive_steps=None, 
                   starting_T='default', ending_T='default', scale=1):

  """Create a Predictor-Corrector (PC) sampler.
  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer. -> not used anymore
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                          sde=sde,
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  
  corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                          sde=sde,
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=c_steps)

  def pc_sampler(model, x=None, show_evolution=False):
    """ The PC sampler funciton.
    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """
    
    #declare non-local variables
    nonlocal scale
    nonlocal starting_T
    nonlocal ending_T
    nonlocal adaptive_steps
    nonlocal p_steps
    nonlocal c_steps

    if show_evolution:
      evolution = []
    
    if starting_T == 'default':
        starting_T = sde.T
    
    if ending_T == 'default':
        ending_T = eps

    with torch.no_grad():
        #Generate the initial sample
        if x is None:
            #if x is not provided, then we generate the random seed of the approximation coefficient at the deepest scale
            x = {'a%d' % scale: sde['a%d' % scale].prior_sampling(shape, starting_T).to(model.device).type(torch.float32)}
        else:
            #if x is provided, then we generate the the random seed of the detail coefficient at the provided scale
            detail_shape = derive_detail_coeff_shape(shape, x.shape)
            detail_coeff = sde['d%d' % scale].prior_sampling(detail_shape, starting_T).to(model.device).type(torch.float32)
            x['d%d' % scale] = detail_coeff

        if adaptive_steps is None:
            total_discrete_points = p_steps+1
            timesteps = torch.linspace(starting_T, ending_T, total_discrete_points, device=model.device)
        else:
            timesteps = adaptive_steps.to(model.device)
            p_steps = timesteps.size(0) - 1
        
        for i in tqdm(range(p_steps)):
            t = timesteps[i]
            vec_t = torch.ones(shape[0], device=t.device) * t
            x, x_mean = corrector_update_fn(x=x, t=vec_t, model=model)
            x, x_mean = predictor_update_fn(x=x, t=vec_t, model=model, discretisation=timesteps)
            
            if show_evolution:
                evolution.append(x.cpu())

        samples = x_mean if denoise else x

    if show_evolution:
        sampling_info = {'evolution': torch.stack(evolution), 
                            'times':timesteps, 'steps': p_steps}
        return samples, sampling_info
    else:
        sampling_info = {'times':timesteps, 'steps': p_steps}
        return samples, sampling_info

  return pc_sampler


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous, discretisation):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, conditional=False, train=False, continuous=continuous)

  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow, discretisation)
  return predictor_obj.update_fn(x, t)

def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper that configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, conditional=False, train=False, continuous=continuous)

  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x, t)

def derive_detail_coeff_shape(current_shape, deeper_shape):
    detail_coeff_shape = []
    for dim_current, dim_deeper in zip(current_shape, deeper_shape):
        if dim_current == dim_deeper:
            detail_coeff_shape.append(dim_current)
        else:
            detail_coeff_shape.append(dim_current - dim_deeper)

    return tuple(detail_coeff_shape)