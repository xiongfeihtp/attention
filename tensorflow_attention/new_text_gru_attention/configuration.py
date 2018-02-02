# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image-to-text model and training configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
    # config
    self.embedding_file="/data/vectors.txt", "vector file"


    self.rnn_size=201
    self.num_rnn_layers=1
    self.embedding_size=300
    self.attention_dim=100
    self.sequence_len=80
    self.sequence_num=200
    self.num_classes=14
    self.dropout=0.5


    self.max_grad_norm=5
    self.init_scale=0.1
    self.batch_size=32
    self.lr=0.1
    self.lr_decay=0.6
    self.epoches=100
    self.max_decay_epoch=30
    self.evaluate_every=1000
    self.l2_reg_lambda=0.01
    self.optimizer = "Adam"

    # Approximate number of values per input shard. Used to ensure sufficient
    # mixing between shards in training.
    self.values_per_input_shard=None
    # Minimum number of shards to keep in the input queue.
    self.num_examples_per_epoch=None

    self.input_queue_capacity_factor=2
    # Number of threads for prefetching SequenceExample protos.
    self.num_input_reader_threads=1
    # Number of threads for image preprocessing. Should be a multiple of 2.
    self.num_preprocess_threads = 4
    # How many model checkpoints to keep.
    self.max_checkpoints_to_keep = 5

    self.sample_feature='sample'
    self.abel_feature='label'
    self.sentences_length_feature='lengths'

    self.vocab=None
    self.input_file_pattern=None

