from pytorch_lightning.callbacks import Callback
from utils import plot

class VisualisationCallback(Callback):

    def on_train_start(self, trainer, pl_module):
        self.visualise(pl_module)

    def on_epoch_end(self,trainer, pl_module):
        self.visualise(pl_module)

    
    def visualise(self, pl_module):
        # log sampled images
        if pl_module.current_epoch % 100 == 0:
            samples = pl_module.sample()
            samples_np =  samples.cpu().numpy()
            image = plot(samples_np[:,0],samples_np[:,1], 'samples epoch: ' + str(pl_module.current_epoch))
            pl_module.logger.experiment.add_image('samples', image, pl_module.current_epoch)