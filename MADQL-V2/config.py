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
CELL_NUMBER = 1
R_MIN = 2.
R_MAX = 20.
INFERENCE_PENALTY_ALPHA = 0.2

# memory pool
MP_MAX_SIZE = 10000
BATCH_SIZE = 512

# DRL hyper-parameter
TOTAL_TIME_SLOT = 1000
LEARNING_RATE = 1e-6                # optimizer learning rate
REG_BETA = 0.2                      # regularization factor
T_STEP = 128                        # update DQN parameter
GAMMA = 0.3                         # Q-value penalty
EPSILON = 0.5                       # Greedy-Epsilon
DECREASE_FACTOR = 0.999             # Decrease epsilon by EPSILON * DECREASE_FACTOR
EVAL_TIMES = 10
HIDDEN_LAYERS = [2048, 2048, 2048]
INPUT_LAYER = 576
OUTPUT_LAYER = 729
PRINT_SLOT = 20                     # print log every PRINT_SLOT

# CNN hyper-parameter


# storage path
MODEL_PATH = "C:\MADQL\model-v2\model.pth"
DATA_PATH = "./simulation_data/data"


if __name__ == "__main__":
    # test generate action list
    print(POWER_LIST)
    print(BEAMFORMER_LIST)
