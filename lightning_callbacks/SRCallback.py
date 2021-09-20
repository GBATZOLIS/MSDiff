from . import utils
import torch
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid
import numpy as np
from torch.nn import Upsample

#make LR-> NNinterpolated, SR, GT appear in this order.
@utils.register_callback(name='bicubic_SR')
class ConditionalHaarMultiScaleVisualizationCallback(Callback):
    def __init__(self, show_evolution=False):
        super().__init__()
        self.show_evolution = show_evolution
        self.upsample_fn = Upsample(scale_factor=2, mode='nearest').to('cpu')
    
    def visualise_conditional_sample(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        y, x = batch
        orig_x = x.clone().cpu()
            
        sampled_images, _ = pl_module.sample(y, self.show_evolution)
        sampled_images = sampled_images.to('cpu')
        upsampled_y = self.upsample_fn(y.to('cpu'))
        super_batch = torch.cat([normalise_per_image(upsampled_y), normalise_per_image(sampled_images), normalise_per_image(orig_x)], dim=-1)

        image_grid = make_grid(super_batch, nrow=int(np.sqrt(super_batch.size(0))))
        pl_module.logger.experiment.add_image('samples_batch_%d_epoch_%d' % (batch_idx, pl_module.current_epoch), image_grid, pl_module.current_epoch)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx==0 and pl_module.current_epoch % 3 == 0:
            self.visualise_conditional_sample(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.visualise_conditional_sample(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)