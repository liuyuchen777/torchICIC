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


def softmax_to_index2(y_predict):
    # power
    p_config_pred = np.zeros(shape=(y_predict.shape[0], bs_num), dtype=int)
    for bs in range(bs_num):
        each_power = y_predict[:, bs * len(P_cb):(bs + 1) * len(P_cb)]
        each_power = np.argmax(each_power, axis=1)
        p_config_pred[:, bs] = each_power
    # beam
    b_config_pred = np.zeros(shape=(y_predict.shape[0], bs_num), dtype=int)
    for bs in range(bs_num):
        each_beam = y_predict[:, bs_num * len(P_cb) + bs * len(precoding_matrices):bs_num * len(P_cb) + (bs + 1) * len(
            precoding_matrices)]
        each_beam = np.argmax(each_beam, axis=1)
        b_config_pred[:, bs] = each_beam

    return p_config_pred, b_config_pred


if __name__ == "__main__":
    # 0) simple load model
    model = torch.load("../model/2021-04-25_18-46-21.pth")
    model.eval()

    # 1) load test data
    data = np.load("../data/test_data_3d/greedy_capacity_b_p_ES_test.npz")
    X = data['X']   # 1000 * 80
    Y = data['Y']   # 1000 * 120
    G_all = data['G_all']
    """

    data = np.load("../data/train_data_3d/greedy_capacity_b_p_ES_train.npz")
    X = data['X'][:1600, :]
    Y = data['Y'][:1600, :]
    G_all = data['G_all'][:1600, :]
    
    """

    # 3) use ES
    # 3-1) convert Y to signal_pow and precoding_index
    p_config, b_config = softmax_to_index(Y)
    signal_pow = np.zeros(p_config.shape, dtype=np.float32)
    for i in range(p_config.shape[0]):
        for j in range(p_config.shape[1]):
            signal_pow[i][j] = P_cb[p_config[i][j]]
    # 3-2) calculate ES system capacity
    C_ES = np.zeros(Y.shape[0], dtype=np.float32)
    for i in range(Y.shape[0]):
        C_ES[i] = system_capacity(G_all[i], signal_pow[i], b_config[i])

    # 4) use SDL model to predict Y_predict
    # 4-1) convert X to tensor and predict
    X_in = torch.tensor(X, dtype=torch.float32)
    Y_predict = model(X_in)
    # 4-2) convert Y_predict (1000 * 39) to p_config, b_config
    y_predict = Y_predict.detach().numpy()
    p_config_pred, b_config_pred = softmax_to_index2(y_predict)
    signal_pow_pred = np.zeros(p_config_pred.shape, dtype=np.float64)
    for i in range(p_config.shape[0]):
        for j in range(p_config.shape[1]):
            signal_pow_pred[i][j] = P_cb[p_config[i][j]]
    # 4-3) calculate SDL system capacity
    C_NN = np.zeros(y_predict.shape[0], dtype=np.float32)
    for i in range(y_predict.shape[0]):
        C_NN[i] = system_capacity(G_all[i], signal_pow_pred[i], b_config_pred[i])
    # 5) random choose
    # 5-1) random generate p_config and b_config
    p_config_rand = np.random.randint(len(P_cb), size=p_config.shape)
    b_config_rand = np.random.randint(len(precoding_matrices), size=b_config.shape)
    signal_pow_rand = np.zeros(p_config.shape, dtype=np.float32)
    # 5-2) signal power
    for i in range(p_config.shape[0]):
        for j in range(p_config.shape[1]):
            signal_pow_rand[i][j] = P_cb[p_config[i][j]]
    # 5-3) calculate capacity
    C_rand = np.zeros(y_predict.shape[0], dtype=np.float32)
    for i in range(y_predict.shape[0]):
        C_rand[i] = system_capacity(G_all[i], signal_pow_rand[i], b_config_rand[i])

    # 6) calculate cdf
    data1 = np.sort(C_ES)
    data2 = np.sort(C_NN)
    data3 = np.sort(C_rand)
    # calculate the proportional values of samples
    p = 1. * np.arange(len(data1)) / (len(data1) - 1)

    plt.plot(data1, p, 'r')
    plt.plot(data2, p, 'b')
    plt.plot(data3, p, 'g')
    plt.xlabel('System Capacity')
    plt.ylabel('CDF')

    plt.show()

    print("done")
