import numpy as np

from const import bs_num, ut_num, n_r, n_t, n_stream, P_cb, check_valid_system, NOISE_POW, precoding_matrices, is_beam_forming
from system_generator import system_generator
from greedy_algorithm import greedy


def cal_cdf(raw_data):
    """
    This function calculate cdf for input data

    Arguments:
    raw_data -- np array, input data in shape of (#data, 1)

    Returns:
    x -- np array in shape of (#data, 1), x of the cdf
    y -- np array in shape of (#data, 1), y of the cdf
    """
    num_data = raw_data.shape[0]
    x = np.sort(raw_data, axis = 0)
    y = (1 + np.arange(num_data)) / num_data
    return x, y


def pair_to_one_hot(pair):
    '''
    This function converts a list indicating pair assignment to a one-hot-coded array.

    Arguments:
    pair -- python list, indicates the BS-UT pairs.
            Each element is a python list of user index assigned to that BS.
    
    Returns:
    one_hot_pair -- np array in shape of (bs_num*ut_num, 1), one-hot-coded array indicating
                    linked bs for each ut
    '''
    one_hot_pair = np.zeros(shape=(bs_num*ut_num, 1))
    for i,p in enumerate(pair):
        one_hot_pair[p[0] * ut_num + i][0] = 1

    return one_hot_pair


def train_data_generator(num_data, mode):
    '''
    This function generate training data
    
    Arguments:
    num_data -- int, number of wanted examples
    mode -- python string, 'capacity' or 'throughput'
    IC -- boolean, if True IC is enabled
    SMIR_after_IC_dB -- after IC, the interference signal of the most power will 
                        have a 10log10(-SMIR_after_IC_dB/10) times power of the desiring recieving signal
    
    Returns:
    X -- np array in shape of (num_data, bs_num*ut_num*len(precoding_matrices)), this is for NNs
    Y -- np array in shape of (num_data, bs_num*len(P_cb)*len(precoding_matrices)), configuration result from greedy algorithm
    '''

    if not check_valid_system():
        print('Invalid global parameters!')
        exit()
    
    Y = np.zeros(shape=(num_data,bs_num*len(P_cb)*len(precoding_matrices)))

    X, G_all = test_data_generator(num_data)
    

    for i in range(num_data):
        print(f"[train_data_generator] iter #{i}")
        one_hot, max_config, max_capacity, capacity = greedy(G_all[i], mode)
        Y[i, :] = np.transpose(one_hot.reshape(bs_num*len(P_cb)*len(precoding_matrices)))

    return X, Y


def test_data_generator(num_data):
    '''
    This function generate testing data
    Arguments:
    num_data -- int, number of wanted examples
    IC -- boolean, if True IC is enabled
    SMIR_after_IC_dB -- after IC, the interference signal of the most power will 
                        have a 10log10(-SMIR_after_IC_dB/10) times power of the desiring recieving signal

    Returns:
    X -- np array in shape of (num_data, bs_num*ut_num*len(precoding_matrices)), this is for NNs
    G_all -- np array of shape (num_data, bs_num, ut_num, len(precoding_matrices), n_r, n_stream), dtype = complex
             all channels of the test data
    '''
    if not check_valid_system():
        print('Invalid global parameters!')
        exit()

    X = np.zeros(shape=(num_data, (bs_num * ut_num + 1), len(precoding_matrices)))
    G_all = np.zeros(shape=(num_data, bs_num, ut_num, len(precoding_matrices), n_r, n_stream), dtype=complex)
    G_ori = np.zeros(shape=(num_data, bs_num, ut_num, n_r, n_t), dtype=complex)
    for i in range(num_data):
        print(f"[test_data_generator] iter #{i}")
        G, _ = system_generator(ifplot = False)
        G_ori[i] = G
        
        if is_beam_forming:
            for j, F in enumerate(precoding_matrices):
                # G in shape of (bs_num, ut_num, n_r, n_t)
                # generate input for NNs
                tmp = G @ F
                X[i, :-1, j] = np.sum(np.abs(tmp)**2, axis=(-1, -2)).flatten()
                X[i, -1, j] = NOISE_POW
                
                G_all[i, :, :, j, :, :] = tmp
            
        else:
            u, s, vh = np.linalg.svd(G) # u @ diag(s) @ vh = G
            tmp = G @ np.swapaxes(vh[:,:,:n_stream,:],-1,-2).conj()
            X[i, :-1, 0] = np.sum(np.abs(tmp)**2, axis=(-1, -2)).flatten()
            X[i, -1, 0] = NOISE_POW
            G_all[i, :, :, 0, :, :] = tmp   

  
        # normalization
        X[i, :, :] /= np.amax(X[i, :, :])
        if np.any(X[i, :, :]==0):
            print("zero occurs")

        X[i, :, :] = np.log10(X[i, :, :])
            
    X = X.reshape((num_data,-1))

    return X, G_all


if __name__ == '__main__':
    num_data = 10
    train_data_generator(num_data, 'capacity')

    
                    

