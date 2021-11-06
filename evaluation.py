#This file is used for evaluation.
#We construct the dataloader and the entire evaluation pipeline.

from torch.utils.data import  Dataset, DataLoader
import glob
import os 


def listdir_nothidden(path, filetype):
    return glob.glob(os.path.join(path, '*.%s' % filetype))

class SynthesizedDataset(Dataset):
    """A template dataset class for you to implement custom datasets."""
    def __init__(self, base_path, snr):
        #base_path: -> samples -> snr_0.150 -> draw_i
        #           -> x_gt
        #           -> y_gt

        sample_paths = {}
        base_sample_path = os.path.join(base_path, 'samples, snr_%.3f' % snr)
        draw_paths = os.listdir(base_sample_path)
        for draw_path in draw_paths:
            sample_paths[draw_path.split('_')[1]] = sorted(listdir_nothidden(os.path.join(base_sample_path, draw_path), 'png'))
        
        y_paths = sorted(listdir_nothidden(os.path.join(base_path, 'y_gt'), 'png'))
        x_paths = sorted(listdir_nothidden(os.path.join(base_path, 'x_gt'), 'png'))