import numpy as np

"""
Use decorator mode to make Config singleton, save repeat compute codebook, improve performance
"""


def dft_matrix(n):
    """
    return a n*n DFT matrix
    """
    dft_i, dft_j = np.meshgrid(np.arange(n), np.arange(n))
    omega = np.exp(- 2 * np.pi * 1J / n)
    w = np.power(omega, dft_i * dft_j) / np.sqrt(n)
    return w


def generateBeamformer(utAntenna):
    beamformer = []
    dftMatrixZ = dft_matrix(utAntenna)
    dftMatrixY = dft_matrix(utAntenna)
    for i in range(utAntenna):
        if i in (1, 2):
            continue
        for j in range(utAntenna):
            precodingMatrix = np.expand_dims(dftMatrixZ[:, i], -1) * np.expand_dims(dftMatrixY[j, :], 0)
            precodingMatrix = precodingMatrix.reshape((-1, 1))
            beamformer.append(precodingMatrix)
    return beamformer


BS_ANTENNA = 16
UT_ANTENNA = 4
BS_HEIGHT = 10.
UT_HEIGHT = 1.5

# power level list
MAX_POWER = 10.  # dBm
POWER_LEVEL = 5
POWER_LIST = []

powerGap = MAX_POWER * 2 / (POWER_LEVEL - 1)
tmpPower = MAX_POWER
for i in range(POWER_LEVEL):
    POWER_LIST.append(tmpPower)
    tmpPower -= powerGap

# beamformer vector list
CODEBOOK_SIZE = 8
BEAMFORMER_LIST = generateBeamformer(UT_ANTENNA)

# wireless channel
ALPHA = 3                           # path loss exponent
SHADOWING_SIGMA = 8                 # unit: dB
GAUSSIAN_SIGMA = 1
# loss, non-loss
NOISE_POWER = -100                  # Gaussian white noise, dBm
PATH_NUMBER = 6                     # LOS path number
RICIAN_FACTOR = 10                  # K_R
# Markov channel change
rho = 0.6425                        # Markov Channel Change

# cellular network
CELL_SIZE = 30.                     # m
CELL_NUMBER = 1
R_MIN = 25.
R_MAX = 30.

# memory pool
MP_MAX_SIZE = 2048
BATCH_SIZE = 256                    # 3-x

# IDQL hyper-parameter
TOTAL_TIME_SLOT = 2000
LEARNING_RATE = 1e-4                # optimizer learning rate
EPSILON = 1                         # Greedy-Epsilon
EPSILON_MIN = 1e-2                  # Min of epsilon value
EPSILON_DECREASE = 1e-4
PRINT_SLOT = 1                      # print log every PRINT_SLOT

# Q-network
INPUT_LAYER = int((3 * CELL_NUMBER) ** 2 * CODEBOOK_SIZE)
OUTPUT_LAYER = CODEBOOK_SIZE * POWER_LEVEL
HIDDEN_LAYER = [1024, 1024, 1024, 1024]

# storage path
MODEL_PATH = "./model/model.pth"
SIMULATION_DATA_PATH = "simulation_data/reward-data-003.txt"
MOBILE_NETWORK_DATA_PATH = "./network_data/network.txt"

if __name__ == "__main__":
    # test generate action list
    print(POWER_LIST)
    print(BEAMFORMER_LIST)
    print(f"len of beamformer： {len(BEAMFORMER_LIST)}")
