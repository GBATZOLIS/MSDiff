from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path
import os

_CALLBACKS = {}
def register_callback(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _CALLBACKS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _CALLBACKS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_callback_by_name(name):
    return _CALLBACKS[name]

def get_callbacks(config, phase='train'):
    callbacks = []

    #check if this works for testing as well.
    if config.training.use_ema:
      callbacks.append(get_callback_by_name('ema')(decay=config.model.ema_rate, ema_device='cpu')) 
    
    if config.training.checkpointing_strategy == 'mixed':
      #save all the checkpoints every K iterations (for post training evaluation)
      checkpoints_path = os.path.join(config.base_log_path, config.experiment_name, 'checkpoint_collection')
      Path(checkpoints_path).mkdir(parents=True, exist_ok=True)
      callbacks.append(ModelCheckpoint(dirpath = checkpoints_path,
                                       filename = '{step}',
                                       monitor = 'step', mode='max',
                                       save_top_k = -1,
                                       every_n_train_steps = config.training.save_every_n_train_steps))
      
      #save only the last checkpoint every M iterations (resuming)
      resume_checkpoint_path = os.path.join(config.base_log_path, config.experiment_name, 'checkpoint_resuming')
      Path(resume_checkpoint_path).mkdir(parents=True, exist_ok=True)
      callbacks.append(ModelCheckpoint(dirpath = resume_checkpoint_path,
                                       filename = 'latest-{step}',
                                       monitor = 'step', mode='max',
                                       save_top_k = 1,
                                       every_n_train_steps = config.training.latest_save_every_n_train_steps))
    
    else:
      #save only the last checkpoint every M iterations (resuming)
      resume_checkpoint_path = os.path.join(config.base_log_path, config.experiment_name, 'checkpoint_resuming')
      Path(resume_checkpoint_path).mkdir(parents=True, exist_ok=True)
      callbacks.append(ModelCheckpoint(dirpath = resume_checkpoint_path,
                                       filename = 'latest-{step}',
                                       monitor = 'step', mode='max',
                                       save_top_k = 1,
                                       every_n_train_steps = config.training.latest_save_every_n_train_steps))

    if phase=='test':
      callbacks.append(get_callback_by_name(config.eval.callback)(config))
    else:
      callbacks.append(get_callback_by_name(config.training.visualization_callback)(config))

    if config.training.lightning_module in ['conditional_decreasing_variance','haar_conditional_decreasing_variance'] :
      callbacks.append(get_callback_by_name('decreasing_variance_configuration')(config))
    else:
      callbacks.append(get_callback_by_name('configuration')())

    return callbacks

  