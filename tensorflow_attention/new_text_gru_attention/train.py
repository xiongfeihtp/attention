from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import configuration
import hierarchy_text_attention

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "./data/output/train-?????-of-00020",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("train_dir", "./train",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_integer("number_of_steps", 1000000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")
tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
    assert FLAGS.train_dir, "--train_dir is required"
    model_config = configuration.ModelConfig()
    model_config.input_file_pattern=FLAGS.input_file_pattern
    # Create training directory.
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        model = hierarchy_text_attention.Hierarchy_text_attention(model_config)
        model.build()
        # Set up the learning rate.
        learning_rate_decay_fn = None
        learning_rate = tf.constant(model_config.lr)

        if model_config.lr_decay > 0:
            num_batches_per_epoch = (model_config.num_examples_per_epoch /
                                     model_config.batch_size)
            decay_steps = int(num_batches_per_epoch *
                              model_config.max_decay_epoch)
            def _learning_rate_decay_fn(learning_rate, global_step):
                return tf.train.exponential_decay(
                    learning_rate,
                    global_step,
                    decay_steps=decay_steps,
                    decay_rate=model_config.lr_decay,
                    staircase=True)
            learning_rate_decay_fn = _learning_rate_decay_fn
        # Set up the training ops.
        train_op = tf.contrib.layers.optimize_loss(
            loss=model.total_loss,
            global_step=model.global_step,
            learning_rate=learning_rate,
            optimizer=model_config.optimizer,
            clip_gradients=model_config.max_grad_norm,
            learning_rate_decay_fn=learning_rate_decay_fn)
        # Set up the Saver for saving and restoring model checkpoints.
        saver = tf.train.Saver(max_to_keep=model_config.max_checkpoints_to_keep)
    # Run training.automatic initialize the threads
    summary_op = tf.summary.merge_all()
    save_summaries_secs = 10
    summary_writer = tf.summary.FileWriter('./log_train')
    tf.contrib.slim.learning.train(
        train_op,
        train_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,
        saver=saver,
        summary_op=summary_op,
        save_summaries_secs=save_summaries_secs,
        summary_writer=summary_writer)
if __name__ == "__main__":
    tf.app.run()
