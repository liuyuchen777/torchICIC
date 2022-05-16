import numpy as np

"""
Use decorator mode to make Config singleton, save repeat compute codebook, improve performance
"""


BS_ANTENNA = 4
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
CODEBOOK_SIZE = 4
BEAMFORMER_LIST = np.zeros(shape=[CODEBOOK_SIZE, BS_ANTENNA], dtype=np.cdouble)
NUMBER_OF_PHASE = 16

for m in range(1, BS_ANTENNA + 1):
    for k in range(1, CODEBOOK_SIZE + 1):
        BEAMFORMER_LIST[k - 1][m - 1] = np.exp(2j * np.pi / NUMBER_OF_PHASE
            * int(m * (k + CODEBOOK_SIZE / 2) % CODEBOOK_SIZE / (CODEBOOK_SIZE / NUMBER_OF_PHASE))) / np.sqrt(BS_ANTENNA)

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
CELL_NUMBER = 7
R_MIN = 2.
R_MAX = 20.

# memory pool
MP_MAX_SIZE = 4096
BATCH_SIZE = 512

# DRL hyper-parameter
TOTAL_TIME_SLOT = 100000
LEARNING_RATE = 1e-4                # optimizer learning rate
REG_BETA = 0.2                      # regularization factor
T_STEP = 256                        # update DQN parameter
EPSILON = 1                         # Greedy-Epsilon
DECREASE_FACTOR = 0.95              # Decrease epsilon by EPSILON * DECREASE_FACTOR
PRINT_SLOT = 256                     # print log every PRINT_SLOT

# storage path
MODEL_PATH = "./model/model.pth"
SIMULATION_DATA_PATH = "./simulation_data/data.txt"
MOBILE_NETWORK_DATA_PATH = "./network_data/network.txt"

if __name__ == "__main__":
    # test generate action list
    print(POWER_LIST)
    print(BEAMFORMER_LIST)
