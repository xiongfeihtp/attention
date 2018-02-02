from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import time
import json
import numpy as np
import tensorflow as tf

import configuration
import hierarchy_text_attention

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "test-?????-of-00004",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("inference_dir", "", "Directory to write event logs.")

tf.flags.DEFINE_integer("num_inference_examples", 10000,
                        "Number of examples for inference.")

tf.logging.set_verbosity(tf.logging.INFO)

def inference_model(sess, model, global_step, summary_writer, summary_op):
    # Log model summaries on a single batch.
    summary_str = sess.run(summary_op)
    summary_writer.add_summary(summary_str, global_step)
    # Compute perplexity over the entire dataset.
    num_inference_batches = int(
        math.ceil(FLAGS.num_inference_examples / model.config.batch_size))
    start_time = time.time()
    value_to_image_id={}
    for i in range(num_inference_batches):
        probabilities_value= sess.run(model.probabilities)
        if not i % 100:  # more efficient
            tf.logging.info("Computed probabilities_value for %d of %d batches.", i + 1,
                            num_inference_batches)
        for item in zip(probabilities_value,model.image_names.eval()):
            value_to_image_id[item[1]]=item[0].tolist()
    inference_time = time.time() - start_time
    # Write the Events file to the eval directory.
    summary_writer.flush()
    tf.logging.info("Finished processing inference at global step %d.inference_time (%.2g sec)",
                    global_step,inference_time)
    with open('value_to_image_id.txt','w') as f:
        json.dump(value_to_image_id,f)

def run_once(model,saver,summary_writer,summary_op):
    model_path = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    if not model_path:
        tf.logging.info("Skipping inference. No checkpoint found in: %s",
                        FLAGS.checkpoint_dir)
        return
    with tf.Session() as sess:
        # Load model from checkpoint.
        tf.logging.info("Loading model from checkpoint: %s", model_path)
        saver.restore(sess, model_path)
        global_step = tf.train.global_step(sess, model.global_step.name)
        tf.logging.info("Successfully loaded %s at global step = %d.",
                        os.path.basename(model_path), global_step)
        # start on cpu for threads runners
        # Start the queue runners.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # Run evaluation on the latest checkpoint.
        try:
            inference_model(
                sess=sess,
                model=model,
                global_step=global_step,
                summary_writer=summary_writer,
                summary_op=summary_op)
        except Exception as e:  # pylint: disable=broad-except
            tf.logging.error("inference failed.")
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def run():
    inference_dir = FLAGS.inference_dir
    if not tf.gfile.IsDirectory(inference_dir):
        tf.logging.info("Creating inference directory: %s", inference_dir)
        tf.gfile.MakeDirs(inference_dir)
    g = tf.Graph()
    with g.as_default():
        # Build the model for evaluation.
        model_config = configuration.ModelConfig()
        model_config.input_file_pattern = FLAGS.input_file_pattern
        model = hierarchy_text_attention.ShowAndTellModel(model_config, mode="inference")
        model.build()
        saver = tf.train.Saver()

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(inference_dir)
        g.finalize()

        tf.logging.info("Starting inference at " + time.strftime(
            "%Y-%m-%d-%H:%M:%S", time.localtime()))
        run_once(model, saver, summary_writer, summary_op)
        tf.logging.info("Competed inference")
def main(unused_argv):
    assert FLAGS.input_file_pattern, "--input_file_pattern is required"
    assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"
    assert FLAGS.inference_dir, "--inference_dir is required"
    run()

if __name__ == "__main__":
    tf.app.run()
