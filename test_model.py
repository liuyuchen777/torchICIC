import torch
import numpy as np
from model_3d_beam_power import *
from hyper_parameter import *
from data_generator.channel_capacity import system_capacity


def softmax_to_index(softmax_vector):
    """
    :param size: the size of data (could be replace)
    :param softmax_vector: [size, 120] data
    :return: [p_config, p_config]
    1. p_config: [size, bs_num] power config
    2. b_config: [size, bs_num] beamforming vector config
    """
    size = softmax_vector.shape[0]
    p_config = np.zeros(shape=(size, bs_num), dtype=int)
    b_config = np.zeros(shape=(size, bs_num), dtype=int)
    for b in range(bs_num):
        temp = softmax_vector[:, b * len(P_cb) * len(precoding_matrices):(b + 1) * len(P_cb) * len(precoding_matrices)]
        p_config[:, b] = np.argmax(temp, axis=1) / len(precoding_matrices)
        b_config[:, b] = np.argmax(temp, axis=1) % len(precoding_matrices)

    return p_config, b_config


def softmax_to_index2(softmax_vector):
    size = softmax_vector.shape[0]


if __name__ == "__main__":
    # 0) simple load model
    model = torch.load("../model/2021-04-14_17-09-54.pth")
    model.eval()

    # 1) load test data
    data = np.load("../data/test_data_3d/greedy_capacity_b_p_ES_test.npz")
    X = data['X']   # 1000 * 80
    Y = data['Y']   # 1000 * 120
    G_all = data['G_all']

    # 3) convert Y to signal_pow and precoding_index
    p_config, b_config = softmax_to_index(Y)
    signal_pow = np.zeros(p_config.shape, dtype=np.float32)
    for i in range(p_config.shape[0]):
        for j in range(p_config.shape[1]):
            signal_pow[i][j] = P_cb[p_config[i][j]]
    # 4) calculate ES system capacity
    C_ES = np.zeros(Y.shape[0], dtype=np.float32)
    for i in range(Y.shape[0]):
        C_ES[i] = system_capacity(G_all[i], signal_pow[i], b_config[i])

    # 4) use NN model to predict Y_predict
    # 4-1) convert X to tensor and predict
    X_in = torch.tensor(X, dtype=torch.float32)
    Y_predict = model(X_in)
    # 4-2) convert Y_predict (1000 * 39) to p_config, b_config


    print("done")
