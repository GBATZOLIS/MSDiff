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

def get_callbacks(config):
    callbacks=[get_callback_by_name('ema')(), \
    get_callback_by_name(config.training.visualization_callback)(show_evolution=config.training.show_evolution)]

    if config.training.lightning_module == 'conditional_decreasing_variance':
      callbacks.append(get_callback_by_name('decreasing_variance_configuration')(
        reduction=config.model.reduction, reach_target_in_epochs=config.model.reach_target_in_epochs,
        starting_transition_iterations = config.model.starting_transition_iterations,
      ))
    else:
      callbacks.append(get_callback_by_name('configuration')())

    return callbacks

  