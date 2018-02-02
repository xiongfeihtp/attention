from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import inputs

class Hierarchy_text_attention(object):
    def __init__(self, config,mode='train'):

        if mode not in ["train", "eval", "inference"]:
            raise ValueError("Unrecognized mode: %s" % mode)

        self.config = config
        self.mode=mode
        self.reader =tf.TFRecordReader()
        self.initializer = tf.random_uniform_initializer(
            minval=-self.config.init_scale,
            maxval=self.config.init_scale)
        self.samples = None
        self.labels=None
        self.sentences_length=None
        # A float32 scalar Tensor; the total loss for the trainer to optimize.
        self.total_loss = None
        # Global step Tensor.
        self.global_step = None

        self.embeddings_path=None
        self.word2idx=self.load_vocab(self.config.vocab)

    def is_training(self):
        """Returns true if the model is built for training mode."""
        return self.mode == "train"

    def build_inputs(self):
        input_queue = inputs.prefetch_input_data(self.reader,
                                                 self.config.input_file_pattern,
                                                 batch_size=self.config.batch_size,
                                                 values_per_shard=self.config.values_per_input_shard,
                                                 input_queue_capacity_factor=self.config.input_queue_capacity_factor,
                                                 num_reader_threads=self.config.num_input_reader_threads,
                                                 is_training=self.is_training())
        assert self.config.num_preprocess_threads % 2 == 0
        data = []
        for thread_id in range(self.config.num_preprocess_threads):
            # thread
            serialized_example = input_queue.dequeue()
            # caption id
            sample, label, sentences_length = inputs.parse_example(
                serialized_example,
                sample_feature=self.config.sample_feature,
                label_feature=self.config.label_feature,
                sentences_length_feature=self.config.sentences_length_feature)
            def data_preprocess(sample, sentences_length):
                # sample,sentences_length
                partition_num = sentences_length.shape[0]
                partition = []
                for i, sentence_length in enumerate(sentences_length):
                    partition.extend(np.ones(sentence_length) * i)
                return tf.dynamic_partition(sample, partition, partition_num)
            sample = data_preprocess(sample, sentences_length)
            data.append([sample, label, sentences_length])
        # mutil threads preprocessing the image
        queue_capacity = (2 * self.config.num_preprocess_threads *
                          self.config.batch_size)
        # pipe for batch data
        # size: samples:[batch_size,sentence_num,sentence_len], labels:[batch_size],sentences_length:[batch_size,sentences_num]
        self.samples, self.labels, self.sentences_length = inputs.batch_with_dynamic_pad(data, batch_size=self.config.batch_size,
                                                                          queue_capacity=queue_capacity)

    def attention_w(self,inputs, attention_size, l2_reg_lambda, name_scope):
        # (batch_size, steps, rnn_size*2)
        sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
        hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer
        # Attention mechanism
        with tf.variable_scope(name_scope):
            W_omega_w = tf.get_variable("W_omega_w",
                                        initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
            b_omega_w = tf.get_variable("b_omega_w", initializer=tf.random_normal([attention_size], stddev=0.1))
            # 随机的
            u_omega_w = tf.get_variable("u_omega_w", initializer=tf.random_normal([attention_size], stddev=0.1))

        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega_w) + tf.reshape(b_omega_w, [1, -1]))
        vu = tf.matmul(v, tf.reshape(u_omega_w, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
        # if l2_reg_lambda > 0:
        #    l2_loss=0
        #    l2_loss += tf.nn.l2_loss(W_omega_w)
        #    l2_loss += tf.nn.l2_loss(b_omega_w)
        #    l2_loss += tf.nn.l2_loss(u_omega_w)
        #    tf.losses.add_loss('losses_s', l2_loss)
        return output


    def attention_s(self,inputs, attention_size, l2_reg_lambda, name_scope):
        sequence_length = inputs.get_shape()[1].value  # the length of sequences processed in the antecedent RNN layer
        hidden_size = inputs.get_shape()[2].value  # hidden size of the RNN layer
        # Attention mechanism
        with tf.variable_scope(name_scope):
            W_omega_s = tf.get_variable("W_omega_s",
                                        initializer=tf.random_normal([hidden_size, attention_size], stddev=0.1))
            b_omega_s = tf.get_variable("b_omega_s", initializer=tf.random_normal([attention_size], stddev=0.1))
            # 随机的
            u_omega_s = tf.get_variable("u_omega_s", initializer=tf.random_normal([attention_size], stddev=0.1))

        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]), W_omega_s) + tf.reshape(b_omega_s, [1, -1]))
        vu = tf.matmul(v, tf.reshape(u_omega_s, [-1, 1]))
        exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
        alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])
        # Output of Bi-RNN is reduced with attention vector
        output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
        # if l2_reg_lambda > 0:
        #    l2_loss=0
        #    l2_loss += tf.nn.l2_loss(W_omega_s)
        #    l2_loss += tf.nn.l2_loss(b_omega_s)
        #    l2_loss += tf.nn.l2_loss(u_omega_s)
        #    tf.losses.add_loss('losses_s', l2_loss)
        return output

    def load_vocab(self,path):
        word2idx = {}
        tf.logging.info("load vocab file to %s", path)
        with tf.gfile.FastGFile(path, "r") as f:
            line = f.readlines()
            for i, word in enumerate(line):
                word = word.strip()
                word2idx[word] = i
        return word2idx

    def load_embedding(self):
        embeddings=np.random.normal(0.00,1.00,[len(self.word2idx),self.config.embedding_size])
        with open(self.embeddings_path,'r') as f:
            lines=f.readlines()
            count=0
            for line in lines[1:]:
                line=line.strip()
                word=line.rstrip().split(" ")[0]
                if (word in self.word2idx):
                    count+=1
                    vec=np.array(line.strip().split(' ')[1:])
                    embeddings[self.word2idx[word]]=vec
        unknown_padding_embedding=np.random.normal(0,0.1,(2,self.config.embedding_size))
        self.embeddings=np.insert(embeddings, [0], unknown_padding_embedding, axis=0)

    def build_model(self):
        document_tensor = tf.unstack(self.samples)
        lengths_tensor=tf.unstack(self.sentences_length)
        output_tensor_list = []
        length_cal=[]
        with tf.name_scope('word'):
            fw_gru_cell = tf.nn.rnn_cell.GRUCell(self.config.rnn_size)
            # if self.is_training and self.keep_prob < 1:
            #    fw_gru_cell =  tf.nn.rnn_cell.DropoutWrapper(
            #        fw_gru_cell, input_keep_prob=self.keep_prob, output_keep_prob = self.keep_prob
            #    )
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_gru_cell] * self.config.num_rnn_layers, state_is_tuple=True)
            # backforward rnn
            bw_gru_cell = tf.nn.rnn_cell.GRUCell(self.config.rnn_size)
            # if self.is_training and self.keep_prob < 1:
            #    bw_gru_cell =  tf.nn.rnn_cell.DropoutWrapper(
            #        bw_gru_cell, input_keep_prob=self.keep_prob, output_keep_prob = self.keep_prob
            #    )
            # 多层的时候才需要在层间增加drop out
            bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_gru_cell] * self.config.num_rnn_layers, state_is_tuple=True)
            # 防止embedding内存过大
            # embedding layer
            X = tf.Variable([0.0])
            self.place = tf.placeholder(tf.float32, shape=(self.embeddings.shape[0], 300))
            embeddings_X = tf.assign(X, self.place, validate_shape=False)
            self.embeddings = tf.Variable(embeddings_X, trainable=True, name="embeddings")
            # 带循环的建图流程

            for i, (sentence_tensor,length_tensor)in enumerate(document_tensor,lengths_tensor):
                if i:
                    reuse = True
                else:
                    reuse = False
                #embedding层最好是在cpu上

                with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
                    inputs = tf.nn.embedding_lookup(self.embeddings, sentence_tensor)#(sentence_num,sentence_len,embedding_dim)

                # dropout input
                if self.is_training and self.config.keep_prob < 1:
                    inputs = tf.nn.dropout(inputs, self.config.keep_prob)
                sequence_length=[i for i in length_tensor if i>0] #(sentence_num,sentence_len)

                length_cal.append(len(sequence_length))

                with tf.variable_scope('bi-gru', reuse=reuse):
                    out_put, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs,
                                                                 sequence_length=sequence_length,
                                                                 dtype=tf.float32)

                out_put = tf.concat(out_put, axis=-1)#(sentence_num,sentence_len,2*hidden_dim)
                output = self.attention_w(out_put, self.config.attention_dim, self.config.l2_reg_lambda, 'word')  # (sentence_number,2*rnn_dim)
                output_tensor_list.append(output)#(document_num,sentence_num,2*hidden_dim)

        with tf.name_scope('sentence'):
            inputs = tf.stack(output_tensor_list, axis=0)  # (bates_size,sentence_number,2*rnn_dim)
            fw_gru_cell = tf.nn.rnn_cell.GRUCell(self.config.rnn_size)
            # if self.is_training and self.keep_prob < 1:
            #    fw_gru_cell =  tf.nn.rnn_cell.DropoutWrapper(
            #        fw_gru_cell, input_keep_prob=self.keep_prob, output_keep_prob = self.keep_prob
            #    )
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_gru_cell] * self.config.num_rnn_layers, state_is_tuple=True)
            # backforward rnn
            bw_gru_cell = tf.nn.rnn_cell.GRUCell(self.config.rnn_size)
            # if self.is_training and self.keep_prob < 1:
            #    bw_gru_cell =  tf.nn.rnn_cell.DropoutWrapper(
            #        bw_gru_cell, input_keep_prob=self.keep_prob, output_keep_prob = self.keep_prob
            #    )
            # 多层的时候才需要在层间增加drop out
            bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_gru_cell] * self.config.num_rnn_layers, state_is_tuple=True)
            # dropout input
            if self.is_training and self.config.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, self.config.keep_prob)

            out_put, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, inputs, sequence_length=length_cal,
                                                         dtype=tf.float32)
            out_put = tf.concat(out_put, axis=-1)
            #(batch_size,sentence_number,2*rnn_dim)
            output = self.attention_s(out_put, self.config.attention_dim, self.config.l2_reg_lambda, 'sentence')  # (batch_size,2*rnn_size)
            #(batch_size,2*rnn_dim)
        # dropout,output
        if self.is_training and self.config.keep_prob < 1:
            output = tf.nn.dropout(output, self.config.keep_prob)

        # Logits.
        with tf.variable_scope("logits") as scope:
            #(batch_size,num_classes)
            self.logits = tf.contrib.layers.fully_connected(
                inputs=output,
                num_outputs=self.config.num_classes,
                activation_fn=None,
                weights_initializer=self.initializer,
                scope=scope)
        with tf.name_scope("loss"):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            batch_loss = tf.reduce_sum(loss)
            tf.losses.add_loss(batch_loss)
            tf.summary.scalar("losses/", batch_loss)

    def build_loss(self):
        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar("losses/total", total_loss)
        self.total_loss = total_loss

    def setup_global_step(self):
        """Sets up the global step Tensor."""
        global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        self.global_step = global_step

    def build(self):
        self.build_inputs()
        self.load_embedding()
        self.build_model()
        self.build_loss()
        self.setup_global_step()
