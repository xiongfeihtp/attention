from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np
import tensorflow as tf
import special_words
FLAGS = tf.flags.FLAGS
#INPUT_FILES="./raw_data/*.txt"
#DATA_DIR="./data_output/"
tf.flags.DEFINE_string("input_files","./raw_data/*.txt",
                       "Comma-separated list of globs matching the input "
                       "files. The format of the input files is assumed to be "
                       "a list of newline-separated sentences, where each "
                       "sentence is already tokenized.")
tf.flags.DEFINE_string("vocab_file", "",
                       "(Optional) existing vocab file. Otherwise, a new vocab "
                       "file is created and written to the output directory. "
                       "The file format is a list of newline-separated words, "
                       "where the word id is the corresponding 0-based index "
                       "in the file.")
tf.flags.DEFINE_string("output_dir", "./data_output/", "Output directory.")
tf.flags.DEFINE_integer("train_output_shards", 20,
                        "Number of output shards for the training set.")
tf.flags.DEFINE_integer("validation_output_shards", 5,
                        "Number of output shards for the validation set.")
tf.flags.DEFINE_integer("validation_ratio", 0.2,
                        "Number of output shards for the validation set.")
tf.flags.DEFINE_integer("num_words", 20000,
                        "Number of words to include in the output.")
tf.logging.set_verbosity(tf.logging.INFO)
def _build_vocabulary(input_files):
    """Loads or builds the model vocabulary.

    Args:
      input_files: List of pre-tokenized input .txt files.

    Returns:
      vocab: A dictionary of word to id, A dictionary of label to id
      max_sentences_num,max_sentences_len
    """

    """
    样本格式:
      图文 希望 赛 重庆 站 决赛 轮   天朗 继续 好 发挥 新浪 体育讯 北京 时间 月 日 消
   男子 领先 组 球员 关 天朗 大力 开球 希望 赛 重庆 站 决赛 轮 球员 全部 出发#女子
   球手 古国 燕杆 优势 领先 争冠 已 无 悬念 亚军 之争 获奖 成为 女选手 主题#男子 
  手 争冠 情形 不明朗 岁 小将 关 天朗 黑马 球手 赵雄 并驾齐驱 闯入 三甲 吴土轩 落
   一杆 精彩 赛事 继续 重庆 邦 高尔夫球场 举行#图为 希望 赛 重庆 站 决赛 轮 精彩 
  间	__label__体育
    """
    if FLAGS.vocab_file:
        tf.logging.info("Loading existing vocab file.")
        vocab = collections.OrderedDict()
        with tf.gfile.GFile(FLAGS.vocab_file, mode="r") as f:
            for i, line in enumerate(f):
                word = line.strip().split("__label__")
                assert word not in vocab, "Attempting to add word twice: %s" % word
                vocab[word] = i
        tf.logging.info("Read vocab of size %d from %s",
                        len(vocab), FLAGS.vocab_file)
        return vocab

    tf.logging.info("Creating vocabulary.")
    num = 0
    wordcount = collections.Counter()
    labelcount = collections.Counter()
    max_sentences_num = 0
    max_sentences_len = 0
    for input_file in input_files:
        tf.logging.info("Processing file: %s", input_file)
        for sentence in tf.gfile.FastGFile(input_file):
            com = sentence.strip().split("__label__")
            word_list = com[0].split('#')
            length = len(word_list)
            if length > max_sentences_num:
                max_sentences_num = length
            for sentence in word_list:
                length=len(sentence.split())
                if length > max_sentences_len:
                    max_sentences_len = length
            word = ' '.join(word_list)
            # word = com[0].replace('#', ' ')
            label = com[1]
            wordcount.update(word.split())
            labelcount.update(label)
            num += 1
            if num % 1000000 == 0:
                tf.logging.info("Processed %d sentences", num)
    tf.logging.info("Processed %d sentences total", num)

    words = list(wordcount.keys())
    freqs = list(wordcount.values())
    # 按照词频进行降序排列
    sorted_indices = np.argsort(freqs)[::-1]
    vocab = collections.OrderedDict()
    vocab[special_words.EOS] = special_words.EOS_ID
    vocab[special_words.UNK] = special_words.UNK_ID

    # 限制词的个数，剔除了低频词，待修改
    for w_id, w_index in enumerate(sorted_indices[0:-2]):
        vocab[words[w_index]] = w_id + 2  # 0: EOS, 1: UNK.
    tf.logging.info("Created vocab with %d words", len(vocab))

    vocab_file = os.path.join(FLAGS.output_dir, "vocab.txt")
    # for orderedDict，key is enough
    with tf.gfile.FastGFile(vocab_file, "w") as f:
        f.write("\n".join(vocab.keys()))
    tf.logging.info("Wrote vocab file to %s", vocab_file)

    word_counts_file = os.path.join(FLAGS.output_dir, "word_counts.txt")
    with tf.gfile.FastGFile(word_counts_file, "w") as f:
        for i in sorted_indices:
            f.write("%s %d\n" % (words[i], freqs[i]))
    tf.logging.info("Wrote word counts file to %s", word_counts_file)

    labels = list(labelcount.keys())
    freqs_label = list(labelcount.values())

    sorted_indices_label = np.argsort(freqs_label)[::-1]
    label_vocab = collections.OrderedDict()
    for label_id, label_index in enumerate(sorted_indices_label):
        label_vocab[labels[label_index]] = label_id
    tf.logging.info("Created vocob with %d labels", len(label_vocab))

    vocab_label_file = os.path.join(FLAGS.output_dir, "vocab_label.txt")
    # for orderedDict，key is enough

    with tf.gfile.FastGFile(vocab_label_file, "w") as f:
        f.write("\n".join(label_vocab.keys()))
    tf.logging.info("Wrote vocab_label file to %s", vocab_label_file)

    label_counts_file = os.path.join(FLAGS.output_dir, "label_counts.txt")
    with tf.gfile.FastGFile(label_counts_file, "w") as f:
        for i in sorted_indices_label:
            f.write("%s %d\n" % (labels[i], freqs_label[i]))
    tf.logging.info("Wrote label counts file to %s", label_counts_file)

    return vocab, label_vocab, max_sentences_num, max_sentences_len


def _int64_feature_list(value):
    """Helper for creating an Int64 Feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(
        value=[v for v in value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def _sentence_to_ids(word_list, vocab, max_sentences_num, max_sentences_len):
    """Helper for converting a sentence (list of words) to a list of ids."""
    document_idx = np.ones((max_sentences_num, max_sentences_len),np.int32)*vocab.get(special_words.EOS_ID)
    # 分别对句子进行编码
    for flag, sentence in enumerate(word_list):
        sent_idx = [vocab.get(word.strip(), vocab.get(special_words.EOS_ID)) for word in sentence.split(" ")]
        # padding
        x_word= np.ones(max_sentences_len, np.int32) * vocab.get(special_words.EOS_ID)
        if len(sent_idx) < max_sentences_len:
            x_word[:len(sent_idx)] = sent_idx
        else:
            x_word = sent_idx[:max_sentences_len]
        document_idx[flag] = x_word
    #返回的是一个np矩阵，可迭代
    return document_idx


def _create_serialized_example(word_list, label, vocab, vocab_label, max_sentences_num, max_sentences_len):
    """Helper for creating a serialized Example proto."""
    example = tf.train.Example(features=tf.train.Features(feature={
        "sample": _int64_feature_list(_sentence_to_ids(word_list, vocab, max_sentences_num, max_sentences_len)),
        "label": _int64_feature(vocab_label.get(label))
    }))
    return example.SerializeToString()


# stats for counter all sentences stats
def _process_input_file(filename, vocab, stats, vocab_label, max_sentences_num, max_sentences_len):
    """Processes the sentences in an input file.
    Args:
      filename: Path to a pre-tokenized input .txt file.
      vocab: A dictionary of word to id.
      stats: A Counter object for statistics.
    Returns:
      processed: A list of serialized Example protos
    """
    tf.logging.info("Processing input file: %s", filename)
    processed = []
    # 统计最大句子数量，和最大句子长度
    for document in tf.gfile.FastGFile(filename):
        stats.update(["sentences_seen"])
        com = document.decode("utf-8").strip().split("__label__")
        word_list = com[0].split("#")
        label = com[1]
        serialized = _create_serialized_example(word_list, label, vocab, vocab_label, max_sentences_num,
                                                max_sentences_len)
        processed.append(serialized)
        stats.update(["sentences_output"])

        sentences_seen = stats["sentences_seen"]
        sentences_output = stats["sentences_output"]

        if sentences_seen and sentences_seen % 100000 == 0:
            tf.logging.info("Processed %d sentences (%d output)", sentences_seen,
                            sentences_output)

        if FLAGS.max_sentences and sentences_output >= FLAGS.max_sentences:
            break
    tf.logging.info("Completed processing file %s", filename)
    return processed


def _write_shard(filename, dataset, indices):
    """Writes a TFRecord shard."""
    with tf.python_io.TFRecordWriter(filename) as writer:
        for j in indices:
            writer.write(dataset[j])


def _write_dataset(name, dataset, indices, num_shards):
    """Writes a sharded TFRecord dataset.

    Args:
      name: Name of the dataset (e.g. "train").
      dataset: List of serialized Example protos.
      indices: List of indices of 'dataset' to be written.
      num_shards: The number of output shards.
    """
    tf.logging.info("Writing dataset %s", name)
    borders = np.int32(np.linspace(0, len(indices), num_shards + 1))
    for i in range(num_shards):
        filename = os.path.join(FLAGS.output_dir, "%s-%.5d-of-%.5d" % (name, i,
                                                                       num_shards))
        shard_indices = indices[borders[i]:borders[i + 1]]
        _write_shard(filename, dataset, shard_indices)
        tf.logging.info("Wrote dataset indices [%d, %d) to output shard %s",
                        borders[i], borders[i + 1], filename)
    tf.logging.info("Finished writing %d sentences in dataset %s.",
                    len(indices), name)

def main(unused_argv):
    if not FLAGS.input_files:
        raise ValueError("--input_files is required.")
    if not FLAGS.output_dir:
        raise ValueError("--output_dir is required.")

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    for pattern in FLAGS.input_files.split(","):
        match = tf.gfile.Glob(FLAGS.input_files)
        if not match:
            raise ValueError("Found no files matching %s" % pattern)
        input_files.extend(match)
    tf.logging.info("Found %d input files.", len(input_files))
    vocab, label_vocab, max_sentences_num, max_sentences_len = _build_vocabulary(input_files)
    # 生成序列化数据
    tf.logging.info("Generating dataset.")
    stats = collections.Counter()
    dataset = []
    for filename in input_files:
        # dataset for different processed serializeds for different filename, stats with dataset information
        dataset.extend(_process_input_file(filename, vocab, stats, label_vocab, max_sentences_num, max_sentences_len))
    sample_number=len(dataset)
    tf.logging.info("Generated dataset with %d sentences.", sample_number)
    for k, v in stats.items():
        tf.logging.info("%s: %d", k, v)
    tf.logging.info("Shuffling dataset.")
    np.random.seed(123)
    shuffled_indices = np.random.permutation(len(dataset))
    split_point=int(FLAGS.validation_ratio * sample_number)
    val_indices = shuffled_indices[:split_point]
    train_indices = shuffled_indices[split_point:]
    # only for a thread
    _write_dataset("train", dataset, train_indices, FLAGS.train_output_shards)
    _write_dataset("validation", dataset, val_indices, FLAGS.validation_output_shards)

if __name__ == "__main__":
    tf.app.run()
