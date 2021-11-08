# ES
import numpy as np
from const import bs_num, P_cb, precoding_matrices
from data_generator.channel_capacity import system_capacity, system_throughput


def convert_num_to_list(x, n):
    """
    This function convert x to a list of number in base n.
    Example: x=6, n=2 returns [1,1,0]

    Arguments:
    x -- int, the orignal number, x >= 0
    n -- base n

    Returns:
    l -- python list
    """
    l = []
    while x > 0:
        l = [x % n] + l
        x = int(x / n)

    l = (bs_num - len(l)) * [0] + l

    return l


def convert_one_hot(l):
    '''
    This function convert list of number in to one-hot-coded array

    Arguments:
    l -- python list, each element is a int indicating the optimal power config

    Returns:
    one_hot -- np array(int), optimal power configuration in shape of (bs_num * len(P_cb), 1), one hot coded for each BS.
    '''
    one_hot = np.zeros(shape=(bs_num * len(P_cb) * len(precoding_matrices), 1), dtype=int)
    cnt = 0
    for i in l:
        one_hot[i + cnt][0] = 1
        cnt += len(P_cb) * len(precoding_matrices)
    return one_hot


def greedy_capacity(G):
    """
    This function calculate a optimal power configuration for each BS by greedy algorithm
    This configuration maximizes system capacity

    Arguments:
    G -- np array, channel matrix of all BSs and UTs in shape of (bs_num, ut_num, len(precoding_matrices), n_r, n_stream), pathloss already included.
    pair -- python list, indicates the BS-UT pairs.
            Each element is a python list of user index assigned to that BS.

    Returns:
    one_hot -- np array(int), optimal power configuration in shape of (bs_num * len(P_cb), 1), one hot coded for each BS.
    max_config -- python list, optimal power and precoding configuration  in shape of (bs_num,)
    max_capacity -- double, the max average capcacity under all possible power and precoding configuration
    capacity -- python list, average capcacity of all possible power and precoding configuration
    """

    capacity = np.zeros(len(P_cb)**bs_num * len(precoding_matrices)**bs_num)
    index = 0
    max_capacity = -1
    for p in range(len(P_cb)**bs_num):
        signal_pow_index = convert_num_to_list(p, len(P_cb))
        signal_pow = [P_cb[m] for m in signal_pow_index]
        for f in range(len(precoding_matrices)**bs_num):
            precoding_index = convert_num_to_list(f, len(precoding_matrices))
            temp_capacity = system_capacity(G, signal_pow, precoding_index)

            capacity[index] = temp_capacity
            index += 1

            if temp_capacity > max_capacity:
                max_capacity = temp_capacity
                max_config = [ind1 * len(precoding_matrices) + ind2 for ind1, ind2 in zip(signal_pow_index, precoding_index)]

    one_hot = convert_one_hot(max_config)
    return one_hot, max_config, np.real(max_capacity), np.real(capacity)


def greedy_throughput(G):
    """
    This function calculate a optimal power configuration for each BS by greedy algorithm
    This configuration maximizes system throughput

    Arguments:
    G -- np array, channel matrix of all BSs and UTs in shape of (bs_num, ut_num, len(precoding_matrices), n_r, n_stream), pathloss already included.
    pair -- python list, indicates the BS-UT pairs.
            Each element is a python list of user index assigned to that BS.

    Returns:
    one_hot -- np array(int), optimal power and precoding configuration
               in shape of (bs_num * len(P_cb) * len(precoding_matrices), 1), one hot coded for each BS.
    max_config -- python list, optimal power and precoding configuration  in shape of (bs_num,)
    max_throughput -- double, the max average throughput under all possible power and precoding configuration
    throughput -- python list, average throughput of all possible power and precoding configuration
    """
    throughput = np.zeros(len(P_cb)**bs_num * len(precoding_matrices)**bs_num)
    index = 0
    max_throughput = -1
    for p in range(len(P_cb)**bs_num):
        signal_pow_index = convert_num_to_list(p, len(P_cb))
        signal_pow = [P_cb[m] for m in signal_pow_index]
        
        for f in range(len(precoding_matrices)**bs_num):
            precoding_index = convert_num_to_list(f, len(precoding_matrices))
            temp_throughput, temp_CQI = system_throughput(G, signal_pow, precoding_index)
            
            throughput[index] = temp_throughput
            index += 1

            if temp_throughput > max_throughput:
                max_throughput = temp_throughput
                max_config = [ind1 * len(precoding_matrices) + ind2 for ind1, ind2 in zip(signal_pow_index, precoding_index)]
                max_CQI = temp_CQI
    
    one_hot = convert_one_hot(max_config)
    return one_hot, max_config, max_throughput, throughput


def greedy(G, mode):
    """
    This function is a wrapper of greedy_capacity and greedy_throughput

    Arguments:
    mode -- python string, 'capacity' or 'throughput'
    """
    
    if mode == 'capacity':
        return greedy_capacity(G)
    elif mode == 'throughput':
        return greedy_throughput(G)
    else:
        print("ERROR: invalid greedy mode")


if __name__ == '__main__':
    G_real = np.random.normal(0.0, 1.0, (3,3,16,4,2))
    G_image = 1j * np.random.normal(0.0, 1.0, (3,3,16,4,2))
    G = G_real + G_image
    greedy_throughput(G, [[0], [1], [2]])

