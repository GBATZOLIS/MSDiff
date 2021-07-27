import torch
import tensorflow as tf
import os
import logging
from matplotlib import pyplot as plt
import io
import PIL
import torchvision.transforms as transforms

def plot(x, y, title):
        fig = plt.figure()
        plt.title(title)
        plt.scatter(x, y)
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = transforms.ToTensor()(image)
        plt.close()
        return image


def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    if 'epoch' in loaded_state.keys():
      state['epoch'] = loaded_state['epoch']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step'],
    'epoch': state['epoch']
  }
  torch.save(saved_state, ckpt_dir)