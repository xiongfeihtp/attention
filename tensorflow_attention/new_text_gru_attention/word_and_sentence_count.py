from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import numpy as np
import tensorflow as tf
import json
from tqdm import tqdm

FLAGS = tf.flags.FLAGS
# INPUT_FILES="./raw_data/*.txt"
# DATA_DIR="./data_output/"
tf.flags.DEFINE_string("input_files", "./raw_data/*.txt",
                       "Comma-separated list of globs matching the input "
                       "files. The format of the input files is assumed to be "
                       "a list of newline-separated sentences, where each "
                       "sentence is already tokenized.")

tf.flags.DEFINE_string("output_dir", "./data_output/", "Output directory.")

tf.logging.set_verbosity(tf.logging.INFO)


def count_document_and_sentence(input_files):
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
    tf.logging.info("begin counter")
    num = 0
    sentence_len_count = collections.Counter()
    document_len_count = collections.Counter()
    for input_file in input_files:
        tf.logging.info("Processing file: %s", input_file)
        for sentence in tqdm(tf.gfile.FastGFile(input_file)):
            com = sentence.strip().split("__label__")
            word_list = com[0].split('#')
            length = len(word_list)
            document_len_count.update(length)
            for sentence in word_list:
                length = len(sentence.split())
                sentence_len_count.update(length)
            num += 1
            if num % 1000000 == 0:
                tf.logging.info("Processed %d sentences", num)
    tf.logging.info("Processed %d sentences total", num)
    with open('./sentence_count_dict.txt', 'w') as f_s, open('./document_count_dict.txt','w') as f_d:
        json.dump(dict(sentence_len_count),f_s)
        json.dump(dict(document_len_count),f_d)
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
    count_document_and_sentence(input_files)


if __name__ == "__main__":
    tf.app.run()
