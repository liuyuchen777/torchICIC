import math
import numpy as np
import struct
import matplotlib.pyplot as plt
import time, os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf

from const import *
from data_generator import test_data_generator


def softmax_to_index(minibatch_size, softmax_vector):
    '''
    This function convert a minibatch of softmax vectors of all bs to
    an array of indecies indicating the max numbers in each vector
    
    Argument:
    minibatch_size -- int, size of a minibatch (= data_num)
    softmax_vector -- np array in shape of (minibatch_size,bs_num*len(P_cb))

    Returns:
    power_config -- np array in shape of (minibatch_size,bs_num), each column is power config for all bs
    '''
    p_config = np.zeros(shape=(minibatch_size,bs_num), dtype=int)
    b_config = np.zeros(shape=(minibatch_size,bs_num), dtype=int)
    for b in range(bs_num):
        temp = softmax_vector[:, b * len(P_cb) * len(precoding_matrices):(b + 1) * len(P_cb) * len(precoding_matrices)]
        p_config[:, b] = np.argmax(temp, axis=1) / len(precoding_matrices)
        b_config[:, b] = np.argmax(temp, axis=1) % len(precoding_matrices)


    return p_config, b_config


def test_model_p(file_path, model_name, X_test, parallel=False):
    '''
    This function test a model for power control
    Arguments:
    file_path -- string indicating the path of the model files
    model_name -- string indicating the name of the model
    X_test -- np array, input of the model
    '''
    data_num = X_test.shape[0]
    Y_softmax = np.zeros((data_num, bs_num * (len(P_cb))))
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        # load model and paras
        saver = tf.train.import_meta_graph(file_path + '/' + model_name + '.meta')
        saver.restore(sess, file_path + '/' + model_name)
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        all_bs_power_softmax = graph.get_operation_by_name('all_softmax').outputs[0]

        start = time.process_time()
        if parallel:
            Y_softmax = sess.run(all_bs_power_softmax, feed_dict={X:X_test})
        else:
            for i in range(data_num):
                Y_softmax[i] = sess.run(all_bs_power_softmax, feed_dict={X:X_test[i].reshape(1, X_test.shape[1])})
    end = time.process_time()    
    p_config = np.zeros(shape=(Y_softmax.shape[0],bs_num), dtype=int)
    b_config = np.zeros(shape=(Y_softmax.shape[0],bs_num), dtype=int)
    for b in range(bs_num):
        temp = Y_softmax[:, b * len(P_cb):(b + 1) * len(P_cb)]
        p_config[:, b] = np.argmax(temp, axis=1) / 1
        b_config[:, b] = np.argmax(temp, axis=1) % 1
        
    return p_config, b_config, str(end-start)[:5]

def test_model_b_p(file_path, model_name, X_test, parallel=False):
    '''
    This function test a model for beam and power control
    Arguments:
    file_path -- string indicating the path of the model files
    model_name -- string indicating the name of the model
    X_test -- np array, input of the model
    '''
    data_num = X_test.shape[0]
    Y_softmax = np.zeros((data_num, bs_num * (len(P_cb) + len(precoding_matrices))))
    tf.reset_default_graph()
    with tf.Session() as sess:
        # load model and paras
        saver = tf.train.import_meta_graph(file_path + '/' + model_name + '.meta')
        saver.restore(sess, file_path + '/' + model_name)
        graph = tf.get_default_graph()
        X = graph.get_tensor_by_name("X:0")
        all_softmax = graph.get_operation_by_name('all_softmax').outputs[0]

        start = time.process_time()
        if parallel:
            Y_softmax = sess.run(all_softmax, feed_dict={X:X_test})
        else:
            for i in range(data_num):
                Y_softmax[i] = sess.run(all_softmax, feed_dict={X:X_test[i].reshape(1, X_test.shape[1])})
    end = time.process_time()
    p_config = np.zeros(shape=(X_test.shape[0],bs_num), dtype=int)
    b_config = np.zeros(shape=(X_test.shape[0],bs_num), dtype=int)

    
    for b in range(bs_num):
        temp = Y_softmax[:, b * len(P_cb):(b + 1) * len(P_cb)]
        p_config[:, b] = np.argmax(temp, axis=1)

        temp = Y_softmax[:, bs_num * len(P_cb) + b * len(precoding_matrices):bs_num * len(P_cb) + (b + 1) * len(precoding_matrices)]
        b_config[:, b] = np.argmax(temp, axis=1)
               
    return p_config, b_config, str(end-start)[:5]
    

if __name__ == '__main__':
    ifplot = True

    # Read data
    data_path = 'train_data/greedy_throughput_b_p.npz'
    file = np.load(data_path)
    G_all = file['G_all']
    Y = file['Y']
    
    
    # Preprocessing
    X = np.sum(np.abs(G_all)**2, axis=(-1, -2))
    X = X.reshape(X.shape[0], -1)
    
    for i in range (X.shape[0]):
        X[i, :] /= np.amax(X[i, :])   
        X[i, :] = np.log10(X[i, :])
    X_test = X[10000:, :]
    p_config, b_config = test_model('./model001-2', 'model001-2', X_test)
    
    
    

    
    power_pdf = np.zeros(shape=(len(P_cb)))
    precoding_pdf = np.zeros(shape=(len(precoding_matrices)))
    for p in p_config:
        power_pdf[p] += 1
    for b in b_config:
        precoding_pdf[b] += 1
    power_pdf /= np.sum(power_pdf)
    precoding_pdf /= np.sum(precoding_pdf)

    if ifplot:
        x = np.arange(len(P_cb))
        plt.bar(x, power_pdf, 0.2)
        plt.text(2, 0.2, 'M=0, SMIR_after_IC_dB=0', style='italic')
        plt.show()

        x = np.arange(len(precoding_matrices))
        plt.bar(x, precoding_pdf, 0.2)
        plt.text(2, 0.2, 'M=0, SMIR_after_IC_dB=0', style='italic')
        plt.show()



