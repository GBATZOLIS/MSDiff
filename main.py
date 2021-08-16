from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import pytorch_lightning as pl
from lightning_models.base import BaseSdeGenerativeModel
from models import ddpm, ncsnv2, fcn
from lightning_data_modules.SyntheticDataset import SyntheticDataModule
from lightning_data_modules.ImageDatasets import ImageDataModule
from callbacks import VisualisationCallback, EMACallback, ImageVisulaizationCallback

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
  config = FLAGS.config
  datamodule = ImageDataModule(config, path=FLAGS.data_path)
  datamodule.setup()
  train_dataloader = datamodule.train_dataloader()

  callbacks=[EMACallback(),
            ImageVisulaizationCallback()]
            
  model = BaseSdeGenerativeModel(config)

  logger = pl.loggers.TensorBoardLogger(FLAGS.log_path)

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

  trainer.fit(model, train_dataloader)

if __name__ == "__main__":
  app.run(main)


