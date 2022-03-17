import abc
import torch
import sde_lib
import numpy as np
from fast_sampling.computation_utils import get_inverse_step_fn

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
  """A decorator for registering predictor classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _PREDICTORS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _PREDICTORS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)

def get_predictor(name):
  return _PREDICTORS[name]

class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""

  def __init__(self, sde, score_fn, probability_flow=False, discretisation=None):
    super().__init__()
    self.sde = sde
    self.probability_flow = probability_flow

    # Compute the reverse SDE/ODE
    if isinstance(sde, dict):
      self.rsde = {}
      for name in sde.keys():
        self.rsde[name] = sde[name].reverse(score_fn, probability_flow)
        #all reverse sdes will share the same score function.
    else:
      self.rsde = sde.reverse(score_fn, probability_flow)

    self.score_fn = score_fn 

    if discretisation is not None:
      self.inverse_step_fn = get_inverse_step_fn(discretisation.cpu().numpy())

  @abc.abstractmethod
  def update_fn(self, x, t):
    """One update of the predictor.
    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.
    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
    pass

@register_predictor(name='ddim')
class DDIMPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False, discretisation=None):
    super().__init__(sde, score_fn, probability_flow, discretisation)
    #assert isinstance(sde, sde_lib.VPSDE), 'ddim sampler is supported only for the VPSDE currently.'

  def multi_scale_update_fn(self, z_t, t):
    dt = torch.tensor(self.inverse_step_fn(t[0].cpu().item())).type_as(t) #-1. / self.rsde.N 
    s = t + dt

    score = self.score_fn(z_t, t)

    z_s = {}
    for name in z_t.keys():
      a_t, sigma_t = self.sde[name].perturbation_coefficients(t[0])
      a_s, sigma_s = self.sde[name].perturbation_coefficients(s[0])

      denoising_value = (sigma_t**2 * score[name] + z_t[name])/a_t
      z_s[name] = sigma_s/sigma_t * z_t[name] + (a_s - sigma_s/sigma_t*a_t) * denoising_value
    
    return z_s, z_s


  def update_fn(self, z_t, t):
    if isinstance(z_t, dict):
      return self.multi_scale_update_fn(z_t, t)
    else:
      #compute the negative timestep
      dt = torch.tensor(self.inverse_step_fn(t[0].cpu().item())).type_as(t) #-1. / self.rsde.N 
      #print(t, dt)

      s = t + dt
      #compute the coefficients
      a_t, sigma_t = self.sde.perturbation_coefficients(t[0])
      a_s, sigma_s = self.sde.perturbation_coefficients(s[0])

      denoising_value = (sigma_t**2 * self.score_fn(z_t, t) + z_t)/a_t
      z_s = sigma_s/sigma_t * z_t + (a_s - sigma_s/sigma_t*a_t)*denoising_value
      return z_s, z_s

@register_predictor(name='heun') #Heun's method (pc-Adams-11-pece)
class PC_Adams_11_Predictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=True, discretisation=None):
    super().__init__(sde, score_fn, probability_flow, discretisation)
    #we implement the PECE method here. This should give us quadratic order of accuracy.

  def f(self, x, t):
    if isinstance(self.sde, dict):
      score = self.score_fn(x, t)
      f = {}
      for name in x.keys():
        f_drift, f_diffusion = self.sde[name].sde(x[name], t)
        r_drift = f_drift - f_diffusion[(..., ) + (None, ) * len(x[name].shape[1:])] ** 2 * score[name] * 0.5
        f[name] = r_drift
      return f
    else:
      drift, diffusion = self.sde.sde(x, t)
      score = self.score_fn(x, t)
      drift = drift - diffusion[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score * 0.5
      return drift

  def predict(self, x, f_0, h):
    if isinstance(self.sde, dict):
      prediction = {}
      for name in x.keys():
        prediction[name] = x[name] + f_0[name] * h
    else:
      prediction = x + f_0 * h

    return prediction
  
  def correct(self, x, f_1, f_0, h):
    if isinstance(self.sde, dict):
      correction = {}
      for name in x.keys():
        correction[name] = x[name] + h/2 * (f_1[name] + f_0[name])
    else:
      correction = x + h/2 * (f_1 + f_0)
    return correction

  def update_fn(self, x, t):
      h = torch.tensor(self.inverse_step_fn(t[0].cpu().item())).type_as(t)
      
      #evaluate
      f_0 = self.f(x, t)
      #predict
      x_1 = self.predict(x, f_0, h)
      #evaluate
      f_1 = self.f(x_1, t+h)
      #correct once
      x_2 = self.correct(x, f_1, f_0, h)

      return x_2, x_2

@register_predictor(name='euler_trapezoidal_s_2_a_0')
class Euler_Trapezoidal_s_2_a_0_Predictor(PC_Adams_11_Predictor):
  def __init__(self, sde, score_fn, probability_flow=True, discretisation=None):
    super().__init__(sde, score_fn, probability_flow, discretisation)
    self.a1 = 0.

  def update_fn(self, x, t):
      h = torch.tensor(self.inverse_step_fn(t[0].cpu().item())).type_as(t)
      
      #evaluate
      f_0 = self.f(x, t)
      #predict
      x_1 = self.predict(x, f_0, h)

      #evaluate
      f_1 = self.f(x_1, t+h)
      #correct
      pece_1 = self.correct(x, f_1, f_0, h)

      #evaluate 
      f_1 = self.f(pece_1, t+h)
      #correct
      pece_2 = self.correct(x, f_1, f_0, h)

      weighted_correction = self.a1 * pece_1 + (1-self.a1) * pece_2
      return weighted_correction, weighted_correction

@register_predictor(name='euler_trapezoidal_s_2_a_7e-1')
class Euler_Trapezoidal_s_2_a_7_Predictor(PC_Adams_11_Predictor):
  def __init__(self, sde, score_fn, probability_flow=True, discretisation=None):
    super().__init__(sde, score_fn, probability_flow, discretisation)
    self.a1 = 0.7

  def update_fn(self, x, t):
      h = torch.tensor(self.inverse_step_fn(t[0].cpu().item())).type_as(t)
      
      #evaluate
      f_0 = self.f(x, t)
      #predict
      x_1 = self.predict(x, f_0, h)

      #evaluate
      f_1 = self.f(x_1, t+h)
      #correct
      pece_1 = self.correct(x, f_1, f_0, h)

      #evaluate 
      f_1 = self.f(pece_1, t+h)
      #correct
      pece_2 = self.correct(x, f_1, f_0, h)

      weighted_correction = self.a1 * pece_1 + (1-self.a1) * pece_2
      return weighted_correction, weighted_correction

@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False, discretisation=None):
    super().__init__(sde, score_fn, probability_flow, discretisation)
    self.probability_flow=probability_flow

  def multi_scale_update_fn(self, x, t):
    dt = torch.tensor(self.inverse_step_fn(t[0].cpu().item())).type_as(t)

    score = self.score_fn(x, t)

    updated_x = {}
    updated_x_mean = {}
    for name in x.keys():
      f_drift, f_diffusion = self.sde[name].sde(x[name], t)

      r_drift = f_drift - f_diffusion[(..., ) + (None, ) * len(x[name].shape[1:])] ** 2 * score[name] * (0.5 if self.probability_flow else 1.)
      r_diffusion = 0. if self.probability_flow else f_diffusion

      x_name_mean = x[name] + r_drift * dt

      if self.probability_flow:
        updated_x[name] = x_name_mean
        updated_x_mean[name] = x_name_mean
      else:
        z = torch.randn_like(x[name])
        updated_name = x_name_mean + r_diffusion[(...,) + (None,) * len(x[name].shape[1:])] * torch.sqrt(-dt) * z
        updated_x[name] = updated_name
        updated_x_mean[name] = x_name_mean
    
    return updated_x, updated_x_mean

  def update_fn(self, x, t):
    if isinstance(x, dict):
      return self.multi_scale_update_fn(x, t)
    else:
      dt = torch.tensor(self.inverse_step_fn(t[0].cpu().item())).type_as(t) #-1. / self.rsde.N
      drift, diffusion = self.rsde.sde(x, t)
      x_mean = x + drift * dt
      
      if self.probability_flow:
        return x_mean, x_mean
      else:
        z = torch.randn_like(x)
        x = x_mean + diffusion[(...,) + (None,) * len(x.shape[1:])] * torch.sqrt(-dt) * z
        return x, x_mean

@register_predictor(name='conditional_euler_maruyama')
class conditionalEulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, y, t):
    dt = -1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, y, t)
    x_mean = x + drift * dt
    x = x_mean + diffusion[(...,) + (None,) * len(x.shape[1:])] * np.sqrt(-dt) * z
    return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False, discretisation=None):
    super().__init__(sde, score_fn, probability_flow, discretisation)

  def multi_scale_update_fn(self, x, t):
    dt = torch.tensor(self.inverse_step_fn(t[0].cpu().item())).type_as(t)
    score = self.score_fn(x, t)
    
    updated_x = {}
    updated_x_mean = {}

    for name in x.keys():
      f_drift, f_diffusion = self.sde[name].sde(x[name], t)
      f_drift_discrete = f_drift * (-dt)
      f_diffusion_discrete = f_diffusion * torch.sqrt(-dt)

      r_drift_discrete = f_drift_discrete - f_diffusion_discrete[(..., ) + (None, ) * len(x[name].shape[1:])] ** 2 * score[name] * (0.5 if self.probability_flow else 1.)
      r_diffusion_discrete = torch.zeros_like(f_diffusion_discrete) if self.probability_flow else f_diffusion_discrete

      x_name_mean = x[name]-r_drift_discrete

      if self.probability_flow:
        updated_x[name] = x_name_mean
        updated_x_mean[name] = x_name_mean
      else:
        z = torch.randn_like(x[name])
        updated_x_name = x_name_mean + r_diffusion_discrete[(...,) + (None,) * len(x[name].shape[1:])] * z

        updated_x[name] = updated_x_name
        updated_x_mean[name] = x_name_mean
    
    return updated_x, updated_x_mean
        

  def update_fn(self, x, t):
    if isinstance(x, dict):
      return self.multi_scale_update_fn(x, t)
    else:
      dt = torch.tensor(self.inverse_step_fn(t[0].cpu().item())).type_as(t)
      score = self.score_fn(x, t)

      f_drift, f_diffusion = self.sde.sde(x, t)
      f_drift_discrete = f_drift * (-dt)
      f_diffusion_discrete = f_diffusion * torch.sqrt(-dt)

      r_drift_discrete = f_drift_discrete - f_diffusion_discrete[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score * (0.5 if self.probability_flow else 1.)
      r_diffusion_discrete = torch.zeros_like(f_diffusion_discrete) if self.probability_flow else f_diffusion_discrete

      x_mean = x - r_drift_discrete

      if self.probability_flow:
        return x_mean, x_mean
      else:
        z = torch.randn_like(x)
        x = x_mean + r_diffusion_discrete[(...,) + (None,) * len(x.shape[1:])] * z
        return x, x_mean
  

@register_predictor(name='conditional_reverse_diffusion')
class conditionalReverseDiffusionPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, y, t):
    f, G = self.rsde.discretize(x, y, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[(...,) + (None,) * len(x.shape[1:])] * z
    return x, x_mean


@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, (sde_lib.VPSDE, sde_lib.VESDE)):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, t)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[(...,) + (None,) * len(x.shape[1:])]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[(...,) + (None,) * len(x.shape[1:])] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, t)
    x_mean = (x + beta[(...,) + (None,) * len(x.shape[1:])] * score) / torch.sqrt(1. - beta)[(...,) + (None,) * len(x.shape[1:])]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[(...,) + (None,) * len(x.shape[1:])] * noise
    return x, x_mean

  def update_fn(self, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)

@register_predictor(name='conditional_ancestral_sampling')
class conditionalAncestralSamplingPredictor(Predictor):
  """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    if not isinstance(sde, (sde_lib.cVESDE, sde_lib.cVPSDE)):
      raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
    assert not probability_flow, "Probability flow not supported by ancestral sampling"

  def vesde_update_fn(self, x, y, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    sigma = sde.discrete_sigmas[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
    score = self.score_fn(x, y, t)
    x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[(...,) + (None,) * len(x.shape[1:])]
    std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
    noise = torch.randn_like(x)
    x = x_mean + std[(...,) + (None,) * len(x.shape[1:])] * noise
    return x, x_mean

  def vpsde_update_fn(self, x, y, t):
    sde = self.sde
    timestep = (t * (sde.N - 1) / sde.T).long()
    beta = sde.discrete_betas.to(t.device)[timestep]
    score = self.score_fn(x, y, t)
    x_mean = (x + beta[(...,) + (None,) * len(x.shape[1:])] * score) / torch.sqrt(1. - beta)[(...,) + (None,) * len(x.shape[1:])]
    noise = torch.randn_like(x)
    x = x_mean + torch.sqrt(beta)[(...,) + (None,) * len(x.shape[1:])] * noise
    return x, x_mean

  def update_fn(self, x, t):
    if isinstance(self.sde, sde_lib.VESDE):
      return self.vesde_update_fn(x, t)
    elif isinstance(self.sde, sde_lib.VPSDE):
      return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, t):
    return x, x

@register_predictor(name='conditional_none')
class NonePredictor(Predictor):
  """An empty predictor that does nothing."""

  def __init__(self, sde, score_fn, probability_flow=False):
    pass

  def update_fn(self, x, y, t):
    return x, x
