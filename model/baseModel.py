"""
说明：这是一个基础函数文件，包含了 slstm 要使用的很多基础操作
日期：2019/04/28
作者：Henry
"""
# -*- coding:utf-8 -*-
import tensorflow as tf

class Baseline(object):
    def __init__(self):
        pass

    def segment1(self, theta, hidden_size):
        W_xf1 = theta[0:hidden_size, 0:hidden_size]
        W_hf1 = theta[hidden_size:3 * hidden_size, 0:hidden_size]
        W_if1 = theta[3 * hidden_size:4 * hidden_size, 0:hidden_size]
        W_df1 = theta[4 * hidden_size:5 * hidden_size, 0:hidden_size]
        b_i1 = theta[-1, 0:hidden_size]
        return [W_xf1, W_hf1, W_if1, W_df1, b_i1]

    def segment2(self, theta, hidden_size):
        W_x_ = theta[0:hidden_size, 0:hidden_size]
        W_h_ = theta[hidden_size:2 * hidden_size, 0:hidden_size]
        b_x = theta[-1, 0:hidden_size]
        return [W_x_, W_h_, b_x]

    def SLSTM_segment1(self, theta, hidden_size):
        W_xf1 = theta[0:hidden_size, 0:hidden_size]
        W_hf1 = theta[hidden_size:3 * hidden_size, 0:hidden_size]
        W_if1 = theta[3 * hidden_size:4 * hidden_size, 0:hidden_size]
        W_df1 = theta[4 * hidden_size:5 * hidden_size, 0:hidden_size]
        W_cf1 = theta[5 * hidden_size:6 * hidden_size, 0:hidden_size]
        b_i1 = theta[-1, 0:hidden_size]
        return [W_xf1, W_hf1, W_if1, W_df1, W_cf1, b_i1]

    def SLSTM_segment2(self, theta, hidden_size):
        W_x_ = theta[0:hidden_size, 0:hidden_size]
        W_h_ = theta[hidden_size:2 * hidden_size, 0:hidden_size]
        W_c_ = theta[2*hidden_size:3 * hidden_size, 0:hidden_size]
        b_x = theta[-1, 0:hidden_size]
        return [W_x_, W_h_, W_c_, b_x]

    def get_hidden_states_before(self, hidden_states, step, shape, hidden_size):
        # padding zeros
        padding = tf.zeros((shape[0], step, hidden_size), dtype=tf.float32)
        # remove last steps
        displaced_hidden_states = hidden_states[:, :-step, :]
        # concat padding
        return tf.concat([padding, displaced_hidden_states], axis=1)
        # return tf.cond(step<=shape[1], lambda: tf.concat([padding, displaced_hidden_states], axis=1),
        # lambda: tf.zeros((shape[0], shape[1], self.config.hidden_size_sum), dtype=tf.float32))

    def get_hidden_states_after(self, hidden_states, step, shape, hidden_size):
        # padding zeros
        padding = tf.zeros((shape[0], step, hidden_size), dtype=tf.float32)
        # remove last steps
        displaced_hidden_states = hidden_states[:, step:, :]
        # concat padding
        return tf.concat([displaced_hidden_states, padding], axis=1)
        # return tf.cond(step<=shape[1], lambda: tf.concat([displaced_hidden_states, padding], axis=1),
        # lambda: tf.zeros((shape[0], shape[1], self.config.hidden_size_sum), dtype=tf.float32))

    def sum_together(self, l):
        combined_state = None
        for tensor in l:
            if combined_state == None:
                combined_state = tensor
            else:
                combined_state = combined_state + tensor
        return combined_state
