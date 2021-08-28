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

def get_callbacks(visualization_callback, show_evolution):
    callbacks=[get_callback_by_name('ema')(), get_callback_by_name('configuration')()]
    callbacks.append(get_callback_by_name(visualization_callback)(show_evolution=show_evolution))
    return callbacks

  