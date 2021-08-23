from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import run_lib

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
    run_lib.train(FLAGS.config, FLAGS.log_path, FLAGS.checkpoint_path)
  elif FLAGS.mode == 'eval':
    raise NotImplementedError

if __name__ == "__main__":
  app.run(main)


