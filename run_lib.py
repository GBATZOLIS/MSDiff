from models import ddpm, ncsnv2, fcn #needed for model registration
import pytorch_lightning as pl

from lightning_callbacks import callbacks, HaarMultiScaleCallback, PairedCallback #needed for callback registration
from lightning_callbacks.utils import get_callbacks

from lightning_data_modules import HaarDecomposedDataset, ImageDatasets, PairedDataset, SyntheticDataset #needed for datamodule registration
from lightning_data_modules.utils import create_lightning_datamodule

from lightning_modules import BaseSdeGenerativeModel, HaarMultiScaleSdeGenerativeModel, ConditionalSdeGenerativeModel #need for lightning module registration
from lightning_modules.utils import create_lightning_module

def train(config, log_path, checkpoint_path):
    DataModule = create_lightning_datamodule(config)
    callbacks = get_callbacks(config.training.visualization_callback, config.training.show_evolution)
    LightningModule = create_lightning_module(config)

    logger = pl.loggers.TensorBoardLogger(log_path, name='lightning_logs')

    if checkpoint_path is not None:
      trainer = pl.Trainer(gpus=config.training.gpus,
                          max_steps=config.training.n_iters, 
                          callbacks=callbacks, 
                          logger = logger,
                          resume_from_checkpoint=checkpoint_path)
    else:  
      trainer = pl.Trainer(gpus=config.training.gpus,
                          max_steps=config.training.n_iters, 
                          logger = logger,
                          callbacks=callbacks
                          )

    trainer.fit(LightningModule, datamodule=DataModule)