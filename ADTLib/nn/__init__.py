# -*- coding: utf-8 -*-
"""
@author: CarlSouthall
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

def BDRNNRestoreAll(X, RestoreFileName, num_layers=3,Truncated=1,n_hidden=50,n_classes=2,cells='tanh'):
    
    tf.reset_default_graph()
    batch_size=0;
    for i in xrange(len(X)):
        if len(X[i]) > batch_size:
            batch_size=len(X[i])
    
    n_input = len(X[0][0][0])
    n_steps = len(X[0][0]) 
    state_len=num_layers 
    
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])
    istate_fw = tf.placeholder("float", [None, (state_len)*n_hidden])
    istate_bw = tf.placeholder("float", [None, (state_len)*n_hidden])
       
    weights = { 'out': tf.Variable(tf.random_normal([n_hidden*2, n_classes]))}    
    biases = { 'out': tf.Variable(tf.random_normal([n_classes]))}
           
    def BiRNN(_X, _istate_fw, _istate_bw, _weights, _biases):
         
         _X = tf.transpose(_X, [1, 0, 2])        
         _X = tf.reshape(_X, [-1, n_input]) 
             
         fw_cell_1 = rnn_cell.BasicRNNCell(n_hidden)
         bw_cell_1 = rnn_cell.BasicRNNCell(n_hidden)
         fw_cell=rnn_cell.MultiRNNCell([fw_cell_1]*num_layers)
         bw_cell=rnn_cell.MultiRNNCell([bw_cell_1]*num_layers)
         _X = tf.split(0, n_steps, _X) 
         seq=np.int32(np.ones(batch_size)*Truncated)
        
         outputs, statefw,statebw = rnn.bidirectional_rnn(fw_cell, bw_cell, _X,
                                                 initial_state_fw=_istate_fw,
                                                 initial_state_bw=_istate_bw,
                                                 sequence_length=seq)
        
         return tf.matmul(outputs[-1], _weights['out']) + _biases['out']
    
    pred = BiRNN(x, istate_fw, istate_bw, weights, biases)
    out=tf.nn.softmax(pred)
    
    init = tf.initialize_all_variables()   
    saver = tf.train.Saver()
    Test=X
    oh=list(np.zeros(len(Test)))
    with tf.Session() as sess:
        
        sess.run(init)
        saver.restore(sess,RestoreFileName) 
        for i in xrange (len(Test)):
            test_len = len(Test[i])
            if test_len != batch_size:
                e=np.zeros((batch_size-test_len,1,len(Test[i][0,0])))
                f=np.concatenate((Test[i],e))
            else: 
                f=Test[i]
            
            o = sess.run(out, feed_dict={x: f,
                                            istate_fw: np.zeros((batch_size, (state_len)*n_hidden)),
                                            istate_bw: np.zeros((batch_size, (state_len)*n_hidden))
                                            })
            oh[i]=o[:test_len]                               
           
       
           
    return oh  
