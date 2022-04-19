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

# generate power list
powerGap = MAX_POWER * 2 / (POWER_LEVEL - 1)
tmpPower = MAX_POWER
for i in range(POWER_LEVEL):
    POWER_LIST.append(tmpPower)
    tmpPower -= powerGap

# beamforming vector list
CODEBOOK_SIZE = 4
BEAMFORMER_LIST = np.zeros(shape=[CODEBOOK_SIZE, BS_ANTENNA], dtype=np.cdouble)

# generate beamformer list
# need to generate three sector of code matrix
N = 16  # number of phases
M = BS_ANTENNA  # number of Antenna
K = CODEBOOK_SIZE  # codebook size
for m in range(1, M + 1):
    for k in range(1, K + 1):
        BEAMFORMER_LIST[k - 1][m - 1] = np.exp(2j * np.pi / N * int(m * (k + K / 2) % K / (K / N))) / np.sqrt(M)

# wireless channel
ALPHA = 3  # path loss exponent
SHADOWING_SIGMA = 8   # db
GAUSSIAN_SIGMA = 1
# loss, non-loss
NOISE_POWER = -100    # Gaussian white noise, dBm
PATH_NUMBER = 6    # LOS path number

# cellular network
CELL_SIZE = 100.  # m
CELL_NUMBER = 7
R_MIN = 15.
R_MAX = 60.
INFERENCE_PENALTY_ALPHA = 0.2

# memory pool
MP_MAX_SIZE = 10000
BATCH_SIZE = 512

# deep learning hyper-parameter
TOTAL_TIME_SLOT = 100
LEARNING_RATE = 1e-4
REG_BETA = 0.
T_STEP = 128
GAMMA = 0.3
EPSILON = 0.3
EVAL_TIMES = 10
HIDDEN_LAYERS = [1024, 1024, 1024]
INPUT_LAYER = 576
OUTPUT_LAYER = 729
PRINT_SLOT = 1
MODEL_PATH = "./model/MADQL.pth"


if __name__ == "__main__":
    # test generate action list
    print(POWER_LIST)
    print(BEAMFORMER_LIST)
