from sampling.predictors import get_predictor, NonePredictor
from sampling.correctors import get_corrector, NoneCorrector
import functools
import torch
from tqdm import tqdm
from models import utils as mutils

def get_conditional_sampling_fn(config, sde, shape, eps):
    predictor = get_predictor(config.sampling.predictor.lower())
    corrector = get_corrector(config.sampling.corrector.lower())
    sampling_fn = get_pc_conditional_sampler(sde=sde, 
                                 shape = shape,
                                 predictor=predictor, 
                                 corrector=corrector, 
                                 snr=config.sampling.snr,
                                 n_steps=config.sampling.n_steps_each, 
                                 probability_flow=config.sampling.probability_flow, 
                                 continuous=config.training.continuous,
                                 denoise=config.sampling.noise_removal, 
                                 eps=eps)
    return sampling_fn

def get_pc_conditional_sampler(sde, shape, predictor, corrector, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3):
  """Create a Predictor-Corrector (PC) sampler.
  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.
  Returns:
    A conditional sampling function that returns samples and the number of function evaluations during sampling.
  """
  # Create predictor & corrector update functions
  predictor_update_fn = functools.partial(conditional_shared_predictor_update_fn,
                                          sde=sde[1],
                                          predictor=predictor,
                                          probability_flow=probability_flow,
                                          continuous=continuous)
  corrector_update_fn = functools.partial(conditional_shared_corrector_update_fn,
                                          sde=sde[1],
                                          corrector=corrector,
                                          continuous=continuous,
                                          snr=snr,
                                          n_steps=n_steps)

  def get_conditional_update_fn(update_fn):
    """Modify the update function of predictor & corrector to incorporate data information."""

    def conditional_update_fn(x, y, t, model):
      with torch.no_grad():
        vec_t = torch.ones(x.shape[0]).to(model.device) * t
        y_mean, y_std = sde[0].marginal_prob(y, vec_t)
        y_perturbed = y_mean + torch.randn_like(y) * y_std[:, None, None, None]
        x, x_mean = update_fn(x=x, y=y_perturbed, t=vec_t, model=model)
        return x, x_mean

    return conditional_update_fn

  predictor_conditional_update_fn = get_conditional_update_fn(predictor_update_fn)
  corrector_conditional_update_fn = get_conditional_update_fn(corrector_update_fn)

  def pc_conditional_sampler(model, y, show_evolution=False):
    """ The PC conditional sampler function.
    Args:
      model: A score model.
    Returns:
      Samples, number of function evaluations.
    """

    def corrections_steps(i):
        return 1

    with torch.no_grad():
      # Initial sample
      x = sde[1].prior_sampling(shape).to(model.device)
      if show_evolution:
        evolution = [x.cpu()]

      timesteps = torch.linspace(sde[1].T, eps, sde[1].N, device=model.device)

      for i in tqdm(range(sde[1].N)):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=model.device) * t
        x, x_mean = predictor_conditional_update_fn(x, y, vec_t, model)

        for _ in range(corrections_steps(i)):
          x, x_mean = corrector_conditional_update_fn(x, y, vec_t, model)
        
        if show_evolution:
          evolution.append(x.cpu())

      if show_evolution:
        sampling_info = {'evolution': torch.stack(evolution)}
        return x_mean if denoise else x, sampling_info
      else:
        return x_mean if denoise else x, {}
        
  return pc_conditional_sampler

def conditional_shared_predictor_update_fn(x, y, t, sde, model, predictor, probability_flow, continuous):
  """A wrapper that configures and returns the update function of predictors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  score_fn = mutils.get_conditional_score_fn(score_fn, target_domain='x')

  if predictor is None:
    # Corrector-only sampler
    predictor_obj = NonePredictor(sde, score_fn, probability_flow)
  else:
    predictor_obj = predictor(sde, score_fn, probability_flow)
  return predictor_obj.update_fn(x, y, t)

def conditional_shared_corrector_update_fn(x, y, t, sde, model, corrector, continuous, snr, n_steps):
  """A wrapper that configures and returns the update function of correctors."""
  score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
  score_fn = mutils.get_conditional_score_fn(score_fn, target_domain='x')

  if corrector is None:
    # Predictor-only sampler
    corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
  else:
    corrector_obj = corrector(sde, score_fn, snr, n_steps)
  return corrector_obj.update_fn(x, y, t)