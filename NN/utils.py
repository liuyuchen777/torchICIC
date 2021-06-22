import logging
import time

import numpy as np
import math
from const import *
from torch.utils.data import Dataset, DataLoader
from hyper_parameter import *


def random_mini_batches(X, Y, mini_batch_size):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (number of examples, input size)
    Y -- true "label" (number of examples, output size)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    # abandon, pytorch have batch tool
    m = X.shape[0]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m / mini_batch_size)
    # number of mini batches of size mini_batch_size in your partitioning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1) * mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def softmax_to_index(mini_batch_size, softmax_vector):
    """
    This function convert a minibatch of softmax vectors of all bs to
    an array of indecies indicating the max numbers in each vector

    Argument:
    minibatch_size -- int, size of a minibatch (= data_num)
    softmax_vector -- np array in shape of (minibatch_size,bs_num*len(P_cb))

    Returns:
    power_config -- np array in shape of (minibatch_size,bs_num), each column is power config for all bs
    """
    p_config = np.zeros(shape=(mini_batch_size, bs_num), dtype=int)
    b_config = np.zeros(shape=(mini_batch_size, bs_num), dtype=int)
    for b in range(bs_num):
        temp = softmax_vector[:, b * len(P_cb) * len(precoding_matrices):(b + 1) * len(P_cb) * len(precoding_matrices)]
        p_config[:, b] = np.argmax(temp, axis=1) / len(precoding_matrices)
        b_config[:, b] = np.argmax(temp, axis=1) % len(precoding_matrices)

    return p_config, b_config


def data_loader(file_path):
    """
    Argument:
    file_path -- train and test data path

    Returns:
    x_train -- train data input
    x_test -- test data input
    y_train -- train data output
    y_test -- test data output
    return values are ndarray
    """
    # load data from file
    data = np.load(file_path)

    # get data
    x_data = data['X']
    y_data = data['Y']
    """
    g_all = data['G_all']
    g_power = np.sum(np.abs(g_all) ** 2, axis=(-1, -2))
    g_power = np.swapaxes(g_power, 1, 2)
    g_power = g_power.reshape((g_all.shape[0], ut_num, -1))
    """

    # divide data to test and train
    x_train = x_data[:16000]
    x_test = x_data[16000:17600]

    power_index_greedy, beam_index_greedy = softmax_to_index(y_data.shape[0], y_data)
    y_data = np.zeros((y_data.shape[0], bs_num * (len(P_cb) + len(precoding_matrices))))

    for i in range(y_data.shape[0]):
        for b in range(bs_num):
            power_index = power_index_greedy[i, b]
            y_data[i, b * len(P_cb) + power_index] = 1

            beam_index = beam_index_greedy[i, b]
            y_data[i, bs_num * len(P_cb) + b * len(precoding_matrices) + beam_index] = 1

    y_train = y_data[:16000]
    y_test = y_data[16000:17600]

    return x_train, x_test, y_train, y_test


class Logger:
    def __init__(self, log_path="./log/"):
        # set path
        log_file_path = log_path + time.strftime("%Y-%m-%d") + ".log"
        logging.basicConfig(filename=log_file_path, format='%(message)s', level=logging.DEBUG)
        # starting write some information
        logging.info("-----------------------NETWORK CONFIG-------------------------------")
        logging.info(f'DATE: {time.strftime("%Y-%m-%d", time.localtime())}, TIME: {time.strftime("%H:%M:%S", time.localtime())}')
        logging.info(f'reg_beta = {reg_beta}, drop_out_rate = {dropout_rate}, learning_rate = {learning_rate}\n'
                     f'num_epochs = {num_epochs}, theta = {theta}, mini_batch_size = {mini_batch_size},\n'
                     f'layer1_unit = {full_con_layer1_unit}, layer2_unit = {full_con_layer2_unit}, mode = {mode}')
        logging.info("-----------------------------END------------------------------------")

    def log(self, log_item):
        print(log_item)
        logging.info(log_item)
