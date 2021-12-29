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
    # Compute the reverse SDE/ODE
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
    assert isinstance(sde, sde_lib.VPSDE), 'ddim sampler is supported only for the VPSDE currently.'

  def compute_coefficients(self, t):
    #this function should be placed inside the SDE classes to generalise the predictor.

    log_mean_coeff = -0.25 * t[0] ** 2 * (self.sde.beta_1 - self.sde.beta_0) - 0.5 * t[0] * self.sde.beta_0
    a_t = torch.exp(log_mean_coeff)
    sigma_t_2 = 1. - torch.exp(2. * log_mean_coeff)
    lambda_t = torch.log(a_t**2/sigma_t_2) #logSNR
    coefficients = {'a': a_t,
                    'sigma_2': sigma_t_2,
                    'lambda': lambda_t}
    return coefficients

  def update_fn(self, x, t):
    #compute the negative timestep
    dt = torch.tensor(self.inverse_step_fn(t[0].cpu().item())).type_as(t) #-1. / self.rsde.N 
    s = t + dt
    #compute the coefficients
    coefficients_t = self.compute_coefficients(t)
    coefficients_s = self.compute_coefficients(s)

    a_ratio = coefficients_s['a']/coefficients_t['a']
    score_multiplier = (1-torch.exp((coefficients_t['lambda'] - coefficients_s['lambda']) / 2))*coefficients_t['sigma_2']
    
    z_t = x
    z_s = a_ratio*(z_t+score_multiplier*self.score_fn(z_t, t))
    return z_s, z_s


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, sde, score_fn, probability_flow=False, discretisation=None):
    super().__init__(sde, score_fn, probability_flow, discretisation)

  def update_fn(self, x, t):
    dt = torch.tensor(self.inverse_step_fn(t[0].cpu().item())).type_as(t) #-1. / self.rsde.N
    z = torch.randn_like(x)
    drift, diffusion = self.rsde.sde(x, t)
    x_mean = x + drift * dt
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
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)

  def update_fn(self, x, t):
    f, G = self.rsde.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[(...,) + (None,) * len(x.shape[1:])] * z
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
