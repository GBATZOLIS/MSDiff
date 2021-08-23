from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import pytorch_lightning as pl
from lightning_modules.utils import get_lightning_module_by_name
from lightning_data_modules.utils import get_lightning_datamodule_by_name
from callbacks.utils import get_callbacks

from models import ddpm, ncsnv2, fcn


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("checkpoint_path", None, "Checkpoint directory.")
flags.DEFINE_string("data_path", None, "Checkpoint directory.")
flags.DEFINE_string("log_path", "./", "Checkpoint directory.")
flags.DEFINE_enum("mode", "train", ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["config", "mode", "log_path"])


def main(argv):
  if FLAGS.mode == 'train':
    config = FLAGS.config

    DataModule = get_lightning_datamodule_by_name(config.data.datamodule)(config)
    callbacks = get_callbacks(config.training.visualization_callback, config.training.show_evolution)
    LightningModule = get_lightning_module_by_name(config.training.lightning_module)

    logger = pl.loggers.TensorBoardLogger(FLAGS.log_path, name='lightning_logs')

    if FLAGS.checkpoint_path is not None:
      trainer = pl.Trainer(gpus=config.training.gpus,
                          max_steps=config.training.n_iters, 
                          callbacks=callbacks, 
                          logger = logger,
                          resume_from_checkpoint=FLAGS.checkpoint_path)
    else:  
      trainer = pl.Trainer(gpus=config.training.gpus,
                          max_steps=config.training.n_iters, 
                          logger = logger,
                          callbacks=callbacks
                          )

    trainer.fit(LightningModule, datamodule=DataModule)

if __name__ == "__main__":
  app.run(main)


