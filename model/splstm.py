# -*- coding:utf-8 -*-
"""
date：2019/04/28
"""
import tensorflow as tf
from model.baseModel import Baseline

basemodel = Baseline()

def SLSTM_segment2(theta, hidden_size):
    W_x_ = theta[0:hidden_size, 0:hidden_size]
    W_h_ = theta[hidden_size:2 * hidden_size, 0:hidden_size]
    b_x = theta[-1, 0:hidden_size]
    return [W_x_, W_h_, b_x]


def splstm_cell(name_scope, config, lengths, initial_hidden_states,
                initial_cell_states, num_layers, domain_Weight, Common, Common_re):
    with tf.name_scope(name_scope):
        hidden_size = config.hidden_size
        Common_re = tf.reshape(Common_re, [-1, hidden_size])
        Common_weight = []
        # parameters for hidden
        weight_len1 = 6 * hidden_size + 1
        Common_weight.append(Common_re[0 * weight_len1:1 * weight_len1, :])
        Common_weight.append(Common_re[1 * weight_len1:2 * weight_len1, :])
        Common_weight.append(Common_re[2 * weight_len1:3 * weight_len1, :])
        Common_weight.append(Common_re[3 * weight_len1:4 * weight_len1, :])
        Common_weight.append(Common_re[4 * weight_len1:5 * weight_len1, :])
        Common_weight.append(Common_re[5 * weight_len1:6 * weight_len1, :])
        Common_weight.append(Common_re[6 * weight_len1:7 * weight_len1, :])
        # parameters for common sentence state
        Common = tf.reshape(Common, [-1, hidden_size])
        weight_len2 = 2 * hidden_size + 1
        Common_weight.append(Common[0 * weight_len2:1 * weight_len2, :])
        Common_weight.append(Common[1 * weight_len2:2 * weight_len2, :])
        Common_weight.append(Common[2 * weight_len2:3 * weight_len2, :])

        domain_Weight = tf.reshape(domain_Weight, [-1, hidden_size])
        domain_weight = []
        domain_weight.append(domain_Weight[0:(2 * hidden_size + 1), :])
        domain_weight.append(domain_Weight[(2 * hidden_size + 1):(4 * hidden_size + 2), :])
        domain_weight.append(domain_Weight[(4 * hidden_size + 2):(6 * hidden_size + 3), :])

        # Word parameters
        Wxf1, Whf1, Wif1, Wdf1, Wcf1, bf1 = basemodel.SLSTM_segment1(Common_weight[0], hidden_size)
        # forget gate for right
        Wxf2, Whf2, Wif2, Wdf2, Wcf2, bf2 = basemodel.SLSTM_segment1(Common_weight[1], hidden_size)
        # forget gate for inital states
        Wxf3, Whf3, Wif3, Wdf3, Wcf3, bf3 = basemodel.SLSTM_segment1(Common_weight[2], hidden_size)
        # forget gate for dummy states
        Wxf4, Whf4, Wif4, Wdf4, Wcf4, bf4 = basemodel.SLSTM_segment1(Common_weight[3], hidden_size)
        # forget gate for common dummy states
        Wxf5, Whf5, Wif5, Wdf5, Wcf5, bf5 = basemodel.SLSTM_segment1(Common_weight[4], hidden_size)
        # input gate for current state
        Wxi, Whi, Wii, Wdi, Wci, bi = basemodel.SLSTM_segment1(Common_weight[5], hidden_size)
        # input gate for output gate
        Wxo, Who, Wio, Wdo, Wco, bo = basemodel.SLSTM_segment1(Common_weight[6], hidden_size)

        # dummy node gated attention parameters
        # input gate for common dummy state
        gated_Whc_c, gated_Wcc_c, gated_bc_c = SLSTM_segment2(Common_weight[7], hidden_size)
        # output gate
        gated_Who_c, gated_Wco_c, gated_bo_c = SLSTM_segment2(Common_weight[8], hidden_size)
        # forget gate for states of word
        gated_Whf_c, gated_Wcf_c, gated_bf_c = SLSTM_segment2(Common_weight[9], hidden_size)

        # Domain parameters
        # domain dummy node gated attention parameters
        gated_Wxd, gated_Whd, gated_bd = SLSTM_segment2(domain_weight[0], hidden_size)
        # output gate
        gated_Wxo, gated_Who, gated_bo = SLSTM_segment2(domain_weight[1], hidden_size)
        # forget gate for states of word
        gated_Wxf, gated_Whf, gated_bf = SLSTM_segment2(domain_weight[2], hidden_size)

    # filters for attention
    mask_softmax_score = tf.cast(tf.sequence_mask(lengths), tf.float32) * 1e25 - 1e25  # [B.L]
    mask_softmax_score_expanded = tf.expand_dims(mask_softmax_score, dim=2)  # [B,L,1]
    # filter invalid steps      # [B,L,1]
    sequence_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lengths), tf.float32), axis=2,
                                   name='sequence_mask')  # [B,L,1]
    # filter embedding states
    initial_hidden_states = initial_hidden_states * sequence_mask
    initial_cell_states = initial_cell_states * sequence_mask
    # record shape of the batch
    shape = tf.shape(initial_hidden_states)  # [B,L,H]

    # initial embedding states
    embedding_hidden_state = tf.reshape(initial_hidden_states, [-1, hidden_size])
    embedding_cell_state = tf.reshape(initial_cell_states, [-1, hidden_size])  # [B*L,H]

    # randomly initialize the states
    if config.random_initialize:  # False
        initial_hidden_states = tf.random_uniform(shape, minval=-0.05, maxval=0.05, dtype=tf.float32, seed=None,
                                                  name=None)
        initial_cell_states = tf.random_uniform(shape, minval=-0.05, maxval=0.05, dtype=tf.float32, seed=None,
                                                name=None)
        # filter it
        initial_hidden_states = initial_hidden_states * sequence_mask
        initial_cell_states = initial_cell_states * sequence_mask

    # inital dummy node states
    dummynode_hidden_states = tf.reduce_mean(initial_hidden_states, axis=1)
    dummynode_cell_states = tf.reduce_mean(initial_cell_states, axis=1)  # [B,H]

    # inital common dummy node states
    dummynode_hidden_states_c = tf.reduce_mean(initial_hidden_states, axis=1)  # [B,H]
    dummynode_cell_states_c = tf.reduce_mean(initial_cell_states, axis=1)  # [B,H]

    for i in range(num_layers):
        # update dummy node states
        # average states
        combined_word_hidden_state = tf.reduce_mean(initial_hidden_states, axis=1)  # [B,H]
        reshaped_hidden_output = tf.reshape(initial_hidden_states, [-1, hidden_size])  # [B*L,H]
        # copy dummy states for computing forget gate
        transformed_dummynode_hidden_states = tf.reshape(
            tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1], 1]), [-1, hidden_size])  # [B*L,H]
        transformed_dummynode_hidden_states_c = tf.reshape(
            tf.tile(tf.expand_dims(dummynode_hidden_states_c, axis=1), [1, shape[1], 1]), [-1, hidden_size])  # [B*L,H]

        ## compute the domain dummy node states
        ## input gate   # [B,H]
        # d==>d    (W*g+W*c+W*h+b)
        gated_d_t = tf.nn.sigmoid(tf.matmul(dummynode_hidden_states, gated_Wxd) +
                                  tf.matmul(combined_word_hidden_state, gated_Whd) + gated_bd)

        # output gate       # [B,H]
        gated_o_t = tf.nn.sigmoid(tf.matmul(dummynode_hidden_states, gated_Wxo) +
                                  tf.matmul(combined_word_hidden_state, gated_Who) + gated_bo)

        # forget gate for hidden states     # [B*L,H]
        gated_f_t = tf.nn.sigmoid(tf.matmul(transformed_dummynode_hidden_states, gated_Wxf) +  # [B*L,H]*[H,H]
                                  tf.matmul(reshaped_hidden_output, gated_Whf) + gated_bf)  # [B*L,H]*[H,H]

        # softmax on each hidden dimension
        reshaped_gated_f_t = tf.reshape(gated_f_t, [shape[0], shape[1], hidden_size]) + mask_softmax_score_expanded

        # [B,L,H] -+- [B,1,H] = [B,L+1,H]
        gated_softmax_scores = tf.nn.softmax(tf.concat([reshaped_gated_f_t,
                                                        tf.expand_dims(gated_d_t, dim=1)], axis=1), dim=1)

        # split the softmax scores
        new_reshaped_gated_f_t = gated_softmax_scores[:, :shape[1], :]
        new_gated_d_t = gated_softmax_scores[:, shape[1]:, :]

        # new dummy states
        dummy_c_t = tf.reduce_sum(new_reshaped_gated_f_t * initial_cell_states, axis=1) \
                    + tf.squeeze(new_gated_d_t, axis=1) * dummynode_cell_states

        dummy_h_t = gated_o_t * tf.nn.tanh(dummy_c_t)

        ################################

        ## input gate   # [B,H]
        # c==>c
        gated_c_c_t = tf.nn.sigmoid(tf.matmul(dummynode_hidden_states_c, gated_Wcc_c) +
                                    tf.matmul(combined_word_hidden_state, gated_Whc_c) + gated_bc_c)

        # output gate       # [B,H]
        gated_o_c_t = tf.nn.sigmoid(tf.matmul(dummynode_hidden_states_c, gated_Wco_c) +
                                    tf.matmul(combined_word_hidden_state, gated_Who_c) + gated_bo_c)

        # forget gate for hidden states     # [B*L,H]
        gated_f_c_t = tf.nn.sigmoid(tf.matmul(transformed_dummynode_hidden_states_c, gated_Wcf_c) +  # [B*L,H]*[H,H]
                                    tf.matmul(reshaped_hidden_output, gated_Whf_c) + gated_bf_c)  # [B*L,H]*[H,H]

        # softmax on each hidden dimension
        reshaped_gated_f_c_t = tf.reshape(gated_f_c_t, [shape[0], shape[1], hidden_size]) \
                               + mask_softmax_score_expanded

        # [B,L,H] -+- [B,1,H] = [B,L+1,H]
        gated_softmax_scores_c = tf.nn.softmax(tf.concat([reshaped_gated_f_c_t,
                                                          tf.expand_dims(gated_c_c_t, dim=1)], axis=1), dim=1)

        # split the softmax scores
        new_reshaped_gated_f_c_t = gated_softmax_scores_c[:, :shape[1], :]
        new_gated_c_c_t = gated_softmax_scores_c[:, shape[1]:, :]

        # new dummy states  (sum f_i*c_i + f_g*c_g)
        dummy_c_c_t = tf.reduce_sum(new_reshaped_gated_f_c_t * initial_cell_states, axis=1) \
                      + tf.squeeze(new_gated_c_c_t, axis=1) * dummynode_cell_states_c
        dummy_h_c_t = gated_o_c_t * tf.nn.tanh(dummy_c_c_t)

        #######################################################################################################

        # update word node states
        # get states before
        initial_hidden_states_before = [
            tf.reshape(basemodel.get_hidden_states_before(initial_hidden_states, step + 1, shape, hidden_size),
                       [-1, hidden_size]) for step in range(config.step)]
        initial_hidden_states_before = basemodel.sum_together(initial_hidden_states_before)  # 对[B,L,H]求和
        initial_hidden_states_after = [
            tf.reshape(basemodel.get_hidden_states_after(initial_hidden_states, step + 1, shape, hidden_size),
                       [-1, hidden_size]) for step in range(config.step)]
        initial_hidden_states_after = basemodel.sum_together(initial_hidden_states_after)
        # get states after
        initial_cell_states_before = [
            tf.reshape(basemodel.get_hidden_states_before(initial_cell_states, step + 1, shape, hidden_size),
                       [-1, hidden_size]) for step in range(config.step)]
        initial_cell_states_before = basemodel.sum_together(initial_cell_states_before)
        initial_cell_states_after = [
            tf.reshape(basemodel.get_hidden_states_after(initial_cell_states, step + 1, shape, hidden_size),
                       [-1, hidden_size]) for step in range(config.step)]
        initial_cell_states_after = basemodel.sum_together(initial_cell_states_after)

        # reshape for matmul
        initial_hidden_states = tf.reshape(initial_hidden_states, [-1, hidden_size])  # [B*L, H]
        initial_cell_states = tf.reshape(initial_cell_states, [-1, hidden_size])  # [B*L, H]

        # concat before and after hidden states
        concat_before_after = tf.concat([initial_hidden_states_before, initial_hidden_states_after], axis=1)

        # copy dummy node states
        transformed_dummynode_hidden_states = tf.reshape(
            tf.tile(tf.expand_dims(dummynode_hidden_states, axis=1), [1, shape[1], 1]), [-1, hidden_size])
        transformed_dummynode_cell_states = tf.reshape(
            tf.tile(tf.expand_dims(dummynode_cell_states, axis=1), [1, shape[1], 1]), [-1, hidden_size])

        # copy common dummy node states
        transformed_dummynode_hidden_states_c = tf.reshape(
            tf.tile(tf.expand_dims(dummynode_hidden_states_c, axis=1), [1, shape[1], 1]), [-1, hidden_size])
        transformed_dummynode_cell_states_c = tf.reshape(
            tf.tile(tf.expand_dims(dummynode_cell_states_c, axis=1), [1, shape[1], 1]), [-1, hidden_size])

        f1_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxf1) + tf.matmul(concat_before_after, Whf1) +
            tf.matmul(embedding_hidden_state, Wif1) + tf.matmul(transformed_dummynode_hidden_states, Wdf1) +
            tf.matmul(transformed_dummynode_hidden_states_c, Wcf1) + bf1)

        f2_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxf2) + tf.matmul(concat_before_after, Whf2) +
            tf.matmul(embedding_hidden_state, Wif2) + tf.matmul(transformed_dummynode_hidden_states, Wdf2) +
            tf.matmul(transformed_dummynode_hidden_states_c, Wcf2) + bf2)

        f3_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxf3) + tf.matmul(concat_before_after, Whf3) +
            tf.matmul(embedding_hidden_state, Wif3) + tf.matmul(transformed_dummynode_hidden_states, Wdf3) +
            tf.matmul(transformed_dummynode_hidden_states_c, Wcf3) + bf3)

        f4_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxf4) + tf.matmul(concat_before_after, Whf4) +
            tf.matmul(embedding_hidden_state, Wif4) + tf.matmul(transformed_dummynode_hidden_states, Wdf4) +
            tf.matmul(transformed_dummynode_hidden_states_c, Wcf4) + bf4)

        f5_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxf5) + tf.matmul(concat_before_after, Whf5) +
            tf.matmul(embedding_hidden_state, Wif5) + tf.matmul(transformed_dummynode_hidden_states, Wdf5) +
            tf.matmul(transformed_dummynode_hidden_states_c, Wcf5) + bf5)

        i_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxi) + tf.matmul(concat_before_after, Whi) +
            tf.matmul(embedding_hidden_state, Wii) + tf.matmul(transformed_dummynode_hidden_states, Wdi) +
            tf.matmul(transformed_dummynode_hidden_states_c, Wci) + bi)

        o_t = tf.nn.sigmoid(
            tf.matmul(initial_hidden_states, Wxo) + tf.matmul(concat_before_after, Who) +
            tf.matmul(embedding_hidden_state, Wio) + tf.matmul(transformed_dummynode_hidden_states, Wdo) +
            tf.matmul(transformed_dummynode_hidden_states_c, Wco) + bo)

        f1_t, f2_t, f3_t, f4_t, f5_t, i_t = tf.expand_dims(f1_t, axis=1), tf.expand_dims(f2_t, axis=1), tf.expand_dims(
            f3_t, axis=1), tf.expand_dims(f4_t, axis=1), tf.expand_dims(f5_t, axis=1), tf.expand_dims(i_t, axis=1)

        six_gates = tf.concat([f1_t, f2_t, f3_t, f4_t, f5_t, i_t], axis=1)
        six_gates = tf.nn.softmax(six_gates, dim=1)
        f1_t, f2_t, f3_t, f4_t, f5_t, i_t = tf.split(six_gates, num_or_size_splits=6, axis=1)  # 切分成6份，从第2维

        f1_t, f2_t, f3_t, f4_t, f5_t, i_t = tf.squeeze(f1_t, axis=1), tf.squeeze(f2_t, axis=1), tf.squeeze(f3_t,
                                                                                                           axis=1), tf.squeeze(
            f4_t, axis=1), tf.squeeze(f5_t, axis=1), tf.squeeze(i_t, axis=1)

        c_t = (f1_t * initial_cell_states_before) + (f2_t * initial_cell_states_after) + \
              (f3_t * embedding_cell_state) + (f4_t * transformed_dummynode_cell_states) + \
              (f5_t * transformed_dummynode_cell_states_c) + (i_t * initial_cell_states)

        h_t = o_t * tf.nn.tanh(c_t)

        # update states
        initial_hidden_states = tf.reshape(h_t, [shape[0], shape[1], hidden_size])
        initial_cell_states = tf.reshape(c_t, [shape[0], shape[1], hidden_size])
        initial_hidden_states = initial_hidden_states * sequence_mask
        initial_cell_states = initial_cell_states * sequence_mask

        dummynode_hidden_states = dummy_h_t
        dummynode_cell_states = dummy_c_t

        dummynode_hidden_states_c = dummy_h_c_t
        dummynode_cell_states_c = dummy_c_c_t

    initial_hidden_states = tf.nn.dropout(initial_hidden_states, config.dropout)
    initial_cell_states = tf.nn.dropout(initial_cell_states, config.dropout)

    return initial_hidden_states, initial_cell_states, dummynode_hidden_states, dummynode_hidden_states_c