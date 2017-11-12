#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 03:46:35 2017

@author: haotianteng
"""

import tensorflow as tf
import numpy as np
#from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMCell
from tensorflow.contrib.rnn import LSTMCell
from utils.lstm import BNLSTMCell
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.rnn.python.ops.rnn import stack_bidirectional_dynamic_rnn
from summary import variable_summaries

def rnn_layers(x,seq_length,training,hidden_num=100,layer_num = 3,class_n = 5):
    cells_fw = list()
    cells_bw = list()
    for i in range(layer_num):
        #cell_fw = BNLSTMCell(hidden_num,training = training)#,training)
        #cell_bw = BNLSTMCell(hidden_num,training = training)#,training)
        cell_fw = LSTMCell(hidden_num)
	cell_bw = LSTMCell(hidden_num)
	cells_fw.append(cell_fw)
        cells_bw.append(cell_bw)
    with tf.variable_scope('BDLSTM_rnn') as scope:
        lasth,_,_=stack_bidirectional_dynamic_rnn(cells_fw = cells_fw,cells_bw=cells_bw,\
                                                inputs = x,sequence_length = seq_length,dtype = tf.float32,scope=scope)
    #shape of lasth [batch_size,max_time,hidden_num*2]
    batch_size = lasth.get_shape().as_list()[0]
    max_time = lasth.get_shape().as_list()[1]
    with tf.variable_scope('rnn_fnn_layer'):
        weight_out = tf.Variable(tf.truncated_normal([2,hidden_num],stddev=np.sqrt(2.0 / (2*hidden_num))),name='weights')
        biases_out = tf.Variable(tf.zeros([hidden_num]),name = 'bias')
        weight_class = tf.Variable(tf.truncated_normal([hidden_num,class_n],stddev=np.sqrt(2.0 / hidden_num)),name = 'weights_class')
        bias_class = tf.Variable(tf.zeros([class_n]),name = 'bias_class')
        lasth_rs = tf.reshape(lasth,[batch_size,max_time,2,hidden_num],name = 'lasth_rs')
        lasth_output = tf.nn.bias_add(tf.reduce_sum(tf.multiply(lasth_rs,weight_out),axis = 2),biases_out,name = 'lasth_bias_add')
        lasth_output_rs = tf.reshape(lasth_output,[batch_size*max_time,hidden_num],name = 'lasto_rs')
        logits = tf.reshape(tf.nn.bias_add(tf.matmul(lasth_output_rs,weight_class),bias_class),[batch_size,max_time,class_n],name = "rnn_logits_rs")
        variable_summaries(weight_class)
        variable_summaries(biases_out)
    return logits

def rnn_layers_one_direction(x,decoder_inputs,seq_length,training,hidden_num=200,layer_num = 3,class_n = 5):
    cells = list()
    for i in range(layer_num):
        cell = BNLSTMCell(hidden_num,training)
        cells.append(cell)
    cell_wrap = tf.contrib.rnn.MultiRNNCell(cells)
    with tf.variable_scope('LSTM_rnn') as scope:
        print(x.shape)
        lasth,encoder_final_state= tf.nn.dynamic_rnn(cell_wrap,x,sequence_length = seq_length,dtype = tf.float32,scope = scope)
    #shape of lasth [batch_size,max_time,hidden_num*2]
    decoder_cell = BNLSTMCell(hidden_num,training)
    #print(encoder_final_state)
    embeddings  = tf.Variable(tf.random_uniform([3, 5], -1.0, 1.0), dtype=tf.float32)
    decoder_emb_inp = tf.nn.embedding_lookup(embeddings, decoder_inputs)
    helper = tf.contrib.seq2seq.TrainingHelper(
           decoder_emb_inp, seq_length
    )
    # Decoder
    projection_layer = layers_core.Dense(
            class_n, use_bias=False
    )
    attention_states = tf.transpose(encoder_final_state[2], [1, 0, 2])
    print(attention_states.shape)
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            200, attention_states,
            memory_sequence_length=seq_length
    )
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=200
    )
    initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=64)
    initial_state = initial_state.clone(cell_state=encoder_final_state[2])
    decoder = tf.contrib.seq2seq.BasicDecoder(
           decoder_cell, helper, initial_state,
           output_layer=projection_layer
    )
    # Dynamic decoding
    #print(decoder.shape)
    outputs, final_context_state, _   = tf.contrib.seq2seq.dynamic_decode(decoder)
    logits = outputs.rnn_output
    #decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)
    #decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn( decoder_cell, decoder_inputs_embedded,initial_state=encoder_final_state[2],dtype=tf.float32, scope="plain_decoder")
    #logits = tf.contrib.layers.linear(decoder_outputs, class_n)
    #print(logits_fake.shape)
    #batch_size = lasth.get_shape().as_list()[0]
    #max_time = lasth.get_shape().as_list()[1]
    #with tf.variable_scope('rnn_fnn_layer'):
    #    weight_class = tf.Variable(tf.truncated_normal([hidden_num,class_n],stddev=np.sqrt(2.0 / hidden_num)),name = 'weights_class')
    #    bias_class = tf.Variable(tf.zeros([class_n]),name = 'bias_class')
    #    lasth_rs = tf.reshape(lasth,[batch_size*max_time,hidden_num],name = 'lasth_rs')
    #    logits = tf.reshape(tf.nn.bias_add(tf.matmul(lasth_rs,weight_class),bias_class),[batch_size,max_time,class_n],name = "rnn_logits_rs")
	#variable_summaries(weight_class)
        #variable_summaries(biases_class)
    #print(logits.shape)
    return logits
