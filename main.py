from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import pytorch_lightning as pl
from lightning_data_modules.utils import get_datamodule_by_name
from lightning_modules.utils import get_lightning_module_by_name
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
flags.mark_flags_as_required(["config", "mode"])


def main(argv):
  if FLAGS.mode == 'train':
    config = FLAGS.config

    DataModule = get_datamodule_by_name(config.data.datamodule)(config)
    callbacks = get_callbacks(config)
    LightningModule = get_lightning_module_by_name(config.lightning_module)

    logger = pl.loggers.TensorBoardLogger(FLAGS.log_path, name='lightning_logs')

    if FLAGS.checkpoint_path is not None:
      trainer = pl.Trainer(gpus=1,
                          max_epochs=int(1e4), 
                          callbacks=callbacks, 
                          logger = logger,
                          resume_from_checkpoint=FLAGS.checkpoint_path)
    else:  
      trainer = pl.Trainer(gpus=1,
                          max_steps=int(4e5), 
                          logger = logger,
                          callbacks=callbacks
                          )

    trainer.fit(LightningModule, datamodule=DataModule)

if __name__ == "__main__":
  app.run(main)


