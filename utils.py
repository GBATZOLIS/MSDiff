import torch
import tensorflow as tf
import os
import logging
from matplotlib import pyplot as plt
import io
import PIL
from torch._C import device
import torchvision.transforms as transforms

def scatter(x, y, **kwargs):
        fig = plt.figure()
        if 'title' in kwargs.keys():
          title = kwargs['title']
          plt.title(title)
        if 'xlim' in kwargs.keys():
          xlim = kwargs['xlim']
          plt.xlim(xlim)
        if 'ylim' in kwargs.keys():  
          ylim = kwargs['ylim']
          plt.ylim(ylim)
        plt.scatter(x, y)
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = transforms.ToTensor()(image)
        plt.close()
        return image

def plot(x, y, title):

        fig = plt.figure()
        plt.title(title)
        plt.plot(x, y)
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = transforms.ToTensor()(image)
        plt.close()
        return image

def create_video(evolution, **kwargs):
        video_tensor = []
        for samples in evolution:
            samples_np =  samples.cpu().numpy()
            image = scatter(samples_np[:,0],samples_np[:,1], **kwargs)
            video_tensor.append(image)
        video_tensor = torch.stack(video_tensor)
        return video_tensor.unsqueeze(0)

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


def compute_grad(f,x,t):
  """
  Args:
      - f - function 
      - x - tensor shape (B, ...) where B is batch size
  Retruns:
      - grads - tensor of gradients for each x
  """
  device = x.device
  x.requires_grad=True
  ftx =f(x,t)
  gradients = torch.autograd.grad(outputs=ftx, inputs=x,
                                  grad_outputs=torch.ones(ftx.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
  gradients = gradients.view(gradients.size(0), -1)
  return gradients

