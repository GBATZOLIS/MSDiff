from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import run_lib

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("checkpoint_path", None, "Checkpoint directory.")
flags.DEFINE_string("data_path", None, "Checkpoint directory.")
flags.DEFINE_string("log_path", "./", "Checkpoint directory.")
flags.DEFINE_enum("mode", "train", \
  
  ["train", "test", \
  "multi_scale_test", \
  "compute_dataset_statistics", \
  'conditional_evaluation_pipeline', \
  'unconditional_evaluation_pipeline', \
  'compute_fast_sampling_scheme',
  'distillation', 
  'compute_lipschitz_constant', 
  'train_multiscale'], \

  "Running mode: train or test")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["config", "mode", "log_path"])


def main(argv):
  if FLAGS.mode == 'train':
    run_lib.train(FLAGS.config, FLAGS.log_path, FLAGS.checkpoint_path)
  elif FLAGS.mode == 'test':
    run_lib.test(FLAGS.config, FLAGS.log_path, FLAGS.checkpoint_path)
  elif FLAGS.mode == 'multi_scale_test':
    run_lib.multi_scale_test(FLAGS.config, FLAGS.log_path)
  elif FLAGS.mode == 'compute_dataset_statistics':
    run_lib.compute_data_stats(FLAGS.config)
  elif FLAGS.mode == 'conditional_evaluation_pipeline':
    run_lib.conditional_evaluation_pipeline(FLAGS.config)
  elif FLAGS.mode == 'unconditional_evaluation_pipeline':
    run_lib.unconditional_evaluation_pipeline(FLAGS.config)
  elif FLAGS.mode == 'compute_fast_sampling_scheme':
    run_lib.compute_fast_sampling_scheme(FLAGS.config, FLAGS.log_path)
  elif FLAGS.mode == 'distillation':
    run_lib.run_distillation(FLAGS.config)
  elif FLAGS.mode == 'compute_lipschitz_constant':
    run_lib.compute_lipschitz_constant(FLAGS.config)
  elif FLAGS.mode == 'train_multiscale':
    run_lib.train_multiscale(FLAGS.config)

if __name__ == "__main__":
  app.run(main)


