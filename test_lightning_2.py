from pytorch_lightning import callbacks
import torch
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from lightning import SdeGenerativeModel
from models import ddpm, ncsnv2, fcn
from SyntheticDataset import SyntheticDataModule
from callbacks import VisualisationCallback

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["config", "mode"])


def main(argv):
  config = FLAGS.config
  datamodule = SyntheticDataModule(config)
  datamodule.setup()
  train_dataloader = datamodule.train_dataloader()

  model = SdeGenerativeModel(config)
  trainer = pl.Trainer(gpus=1,max_steps=int(4e5), callbacks=[VisualisationCallback()])
  trainer.fit(model, train_dataloader)

if __name__ == "__main__":
  app.run(main)


