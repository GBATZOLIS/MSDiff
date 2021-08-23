LIGHTNING_DATA_MODULES = {}
def register_lightning_datamodule(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in LIGHTNING_DATA_MODULES:
      raise ValueError(f'Already registered model with name: {local_name}')
    LIGHTNING_DATA_MODULES[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_lightning_datamodule_by_name(name):
  print(LIGHTNING_DATA_MODULES.keys())
  return LIGHTNING_DATA_MODULES[name]
