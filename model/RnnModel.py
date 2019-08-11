# -*- coding:utf-8 -*-
import tensorflow as tf

def bilstm_layer(initial_hidden_states, config, keep_prob, mask):
    with tf.variable_scope('forward', reuse=False):
        fw_lstm = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0)
        fw_lstm = tf.contrib.rnn.DropoutWrapper(fw_lstm, output_keep_prob=keep_prob)
        print(fw_lstm.name)

    with tf.variable_scope('backward', reuse=False):
        bw_lstm = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0)
        bw_lstm = tf.contrib.rnn.DropoutWrapper(bw_lstm, output_keep_prob=keep_prob)

    # bidirectional rnn
    with tf.variable_scope('bilstm', reuse=False):
        lstm_output = tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, inputs=initial_hidden_states,
                                                      sequence_length=mask, time_major=False, dtype=tf.float32)
        lstm_output = tf.concat(lstm_output[0], 2)
        print(lstm_output.name)

    return lstm_output


def lstm_layer(initial_hidden_states, config, keep_prob, mask):
    with tf.variable_scope('cell'):
        cell_lstm = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0)
        cell_lstm = tf.contrib.rnn.DropoutWrapper(cell_lstm, output_keep_prob=keep_prob)

    # rnn
    with tf.variable_scope('lstm_layer'):
        lstm_output, _ = tf.nn.dynamic_rnn(cell_lstm, inputs=initial_hidden_states, sequence_length=mask,
                                           time_major=False, dtype=tf.float32)
        lstm_output = tf.concat(lstm_output, 2)

    return lstm_output