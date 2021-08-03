import numpy as np

bs_num = 3          # number of BSs
ut_num = 3          # number of UTs
link_num = 3        # number of links

area_length = 30.
bs_height = 10
ut_height = 1.5

# position of base station: 3D, [x, y, z]
bs_pos = np.asarray([
    [area_length / 4, 0., bs_height],
    [area_length / 4, area_length / 2 * np.sqrt(3), bs_height],
    [area_length, area_length / 4 * np.sqrt(3), bs_height]
    ])

# don't know what it is
bs_dir_no_tilt = np.asarray([
    [[1./2, np.sqrt(3)/2, 0.], [-np.sqrt(3)/2, 1./2, 0.], [0., 0., 1.]],
    [[1./2, -np.sqrt(3)/2, 0.], [np.sqrt(3)/2, 1./2, 0.], [0., 0., 1.]],
    [[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]
    ])

bs_tilt = 30. * np.pi / 180.
bs_dir = np.zeros(bs_dir_no_tilt.shape)
for b in range(bs_num):
    bs_dir[b, 0, :2] = bs_dir_no_tilt[b, 0, :2] * np.cos(bs_tilt)
    bs_dir[b, 0, 2] = -bs_dir_no_tilt[b, 2, 2] * np.sin(bs_tilt)

    bs_dir[b, 1] = bs_dir_no_tilt[b, 1]

    bs_dir[b, 2, :2] = bs_dir[b, 0, :2] * np.sin(bs_tilt)
    bs_dir[b, 2, 2] = bs_dir_no_tilt[b, 2, 2] * np.cos(bs_tilt)

n_t_y = 4
n_t_z = 4
n_t = n_t_y * n_t_z     # number of transmitting antenna of one BS
n_r = 4                 # number of receiving antenna of one UT
n_interval = 0.5        # interval between array elements is 0.5λ
n_stream = 1


K_factor = 10
N_path = 6
N_component = 8
max_delay = 32
COMPONENT = 8               # Rayleigh channel component
N_pk = 10                   # number of packet per transmission, during a transmission pathloss doesn't change
P_cb = [-10, -5, 0, 5, 10]  # [important] dBm power code book for BS to choose from

NOISE_POW = 1e-13           # 1e-13


# path loss parameters
path_loss_exponent = 4
log_std = 8	                # log-normal shadowing standard deviation in dB

# CQI_to_throughput list
CQI_to_throughput = [0, 0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758,
                     1.4766, 1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234,
                     5.1152, 5.5547]

if_plot_system_map = True

# switch of beam-forming
is_beam_forming = True


def check_nan(a):
    """
    This function check if there is at least one np.nan in a given np array
    Arguments:
    a -- np array

    Returns:
    ifnan -- if there is at least a nan, return True
             else return False
    """
    return np.any(np.isnan(a))
    # return np.isnan(np.sum(a))


def check_inf(a):
    """
    This function check if there is at least one inf/-inf in a given np array
    Arguments:
    a -- np array

    Returns:
    ifinf -- if there is at least a nan, return True
             else return False
    """
    return np.any(np.isinf(a))


def check_valid_system():
    """
    This function check global parameters to see if they are valid

    Returns:
    valid -- boolean, if system is valid return true, else return false
    """

    valid = True
    if not (bs_num == ut_num and bs_num == link_num):
        valid = False
    if not (bs_pos.shape[0] == bs_dir.shape[0] and bs_pos.shape[0] == bs_num):
        valid = False

    return valid


def dft_matrix(n):
    """
    return a n*n DFT matrix
    """
    dft_i, dft_j = np.meshgrid(np.arange(n), np.arange(n))
    omega = np.exp(- 2 * np.pi * 1J / n)
    w = np.power(omega, dft_i * dft_j) / np.sqrt(n)
    return w


# if use beam-forming, calculate pre-coding matrix
if is_beam_forming:
    precoding_matrices = []
    dft_mtx_z = dft_matrix(n_t_z)
    dft_mtx_y = dft_matrix(n_t_y)
    for i in range(n_t_z):
        if i in (1, 2):
            continue
        for j in range(n_t_y):
            precoding_matrix = np.expand_dims(dft_mtx_z[:, i], -1) * np.expand_dims(dft_mtx_y[j, :], 0)
            precoding_matrix = precoding_matrix.reshape((-1, 1))
            precoding_matrices.append(precoding_matrix)
else:
    precoding_matrices = [-1]
