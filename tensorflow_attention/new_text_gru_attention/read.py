import tensorflow as tf
import numpy as np

def read_queue(filename=None, reader=None, shuffle=False, capacity=100, num_reader_threads=1, batch_size=10):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
    # filename_queue = tf.train.string_input_producer(
    #       data_files, shuffle=shuffle, capacity=16, name="filename_queue")
    if shuffle:
        min_after_dequeue = int(0.6 * capacity)
        values_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_after_dequeue,
            dtypes=[tf.string],
            shapes=[[]],
            name="random_input_queue")
    else:
        values_queue = tf.FIFOQueue(
            capacity=capacity,
            dtypes=[tf.string],
            shapes=[[]],
            name="fifo_input_queue")
    enqueue_ops = []
    for _ in range(num_reader_threads):
        _, value = reader.read(filename_queue)
        enqueue_ops.append(values_queue.enqueue([value]))
    tf.train.queue_runner.add_queue_runner(
        tf.train.queue_runner.QueueRunner(values_queue, enqueue_ops))
    serialized = values_queue.dequeue_many(batch_size)
    #样本serialized
    features = tf.parse_example(
        serialized,
        features={
            'sample': tf.VarLenFeature(dtype=tf.int64),
            'label': tf.VarLenFeature(dtype=tf.int64),
            'lengths': tf.VarLenFeature(dtype=tf.int64)
        }
        # features={
        #     'sample': tf.FixedLenFeature([100], dtype=tf.int64),
        #     'label': tf.FixedLenFeature([100], dtype=tf.int64),
        #     'lengths': tf.FixedLenFeature([100], dtype=tf.int64)
        # }
    )
    def _sparse_to_batch(sparse):
        ids = tf.sparse_tensor_to_dense(sparse)  # Padding with zeroes.
        mask = tf.sparse_to_dense(sparse.indices, sparse.dense_shape,
                                  tf.ones_like(sparse.values, dtype=tf.int32))
        return ids, mask
    a, a_mask = _sparse_to_batch(features['sample'])
    b, b_mask = _sparse_to_batch(features['label'])
    c, c_mask = _sparse_to_batch(features['lengths'])
    # a = features['sample']
    # b = features['label']
    # c = features['lengths']
    return a, a_mask, b, b_mask, c, c_mask


def data_preprocess(sample, sentences_length):
    # sample,sentences_length
    partition_num = sentences_length.get_shape()
    print(partition_num)
    partition = []
    for i in range(partition_num):
        sentence_length = sentences_length[i]
        partition.extend(np.ones(sentence_length) * i)
    # for i, sentence_length in enumerate(sentences_length):
    #     partition.extend(np.ones(sentence_length) * i)
    return tf.dynamic_partition(sample, partition, partition_num)

data = []
# create tensor
a, a_mask, b, b_mask, c, c_mask = read_queue('./validation-00000-of-00005')
print(a_mask.shape)

# sample = data_preprocess(a, c)
# data.append([sample, b, c])
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
tf.train.start_queue_runners(sess=sess)
a_val, a_mask_val, b_val, b_val_mask, c_val, c_val_mask = sess.run([a, a_mask, b, b_mask, c, c_mask])