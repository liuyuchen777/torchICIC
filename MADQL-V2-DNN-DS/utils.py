import json
import logging
from enum import Enum

import time
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde

from config import *

"""
Functions to convert number to dBm/dB (power)
"""


def dBm2num(dB):
    num = (10 ** (dB / 10)) / 1000
    return num


def dB2num(dB):
    num = 10 ** (dB / 10)
    return num


def num2dB(num):
    dB = 10. * np.log10(num)
    return dB


def setLogger(file=True, debug=False):
    if debug:
        logLevel = logging.DEBUG
    else:
        logLevel = logging.INFO
    if file:
        logFilePath = "./log/" + time.strftime("%Y-%m-%d") + ".log"
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%H:%M:%S', level=logLevel,
                            handlers=[logging.FileHandler(logFilePath), logging.StreamHandler()])
    else:
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%H:%M:%S', level=logLevel)
    # initial log
    logging.info("=====================================CONFIG=========================================")
    logging.info(f'START TIME: {time.strftime("%H:%M:%S", time.localtime())}')
    logging.info("-----------------------------------COMMUNICATION------------------------------------")
    logging.info(f'Power Level: {POWER_LEVEL}, Codebook Size: {CODEBOOK_SIZE}, '
                 f'Cell Length: {CELL_SIZE} m, Cell Number: {CELL_NUMBER}')
    logging.info(f'Path Loss Exponent: {ALPHA}, Log-normal Sigma: {SHADOWING_SIGMA} db, '
                 f'Gaussian Sigma: {GAUSSIAN_SIGMA} db')
    logging.info("-----------------------------------------DL------------------------------------------")
    logging.info(f'Batch Size: {BATCH_SIZE}, Learning Rate: {LEARNING_RATE}, Output Layer: {OUTPUT_LAYER}')
    logging.info(f'Epsilon: {EPSILON}, Epsilon Decrease: {EPSILON_DECREASE}, Epsilon Min: {EPSILON_MIN}, Input Layer: {INPUT_LAYER}')
    logging.info(f'Network Hidden Layers: {HIDDEN_LAYER}, Interference Penalty: {INTERFERENCE_PENALTY}')
    # network config information
    logging.info("=========================================END=========================================")


class Algorithm(Enum):
    RANDOM = 1
    MAX_POWER = 2
    MADQL = 3
    CELL_ES = 4


# Interference skip list -> due sector isolation
SKIP_LIST = {
    0: [4, 8],
    1: [11, 12],
    2: [15, 19],
    3: [],
    4: [0, 8],
    5: [18],
    6: [],
    7: [9],
    8: [0, 4],
    9: [7],
    10: [],
    11: [1, 12],
    12: [1, 11],
    13: [],
    14: [16],
    15: [2, 19],
    16: [14],
    17: [],
    18: [5],
    19: [2, 15],
    20: []
}


def generateChannelIndex(transmitterIndex, receiverIndex):
    return f'{transmitterIndex}-{receiverIndex}'


def calCapacity(actions, env):
    capacity = []

    for i in range(len(actions)):
        power = dBm2num(POWER_LIST[actions[i][0]])
        beamformer = BEAMFORMER_LIST[actions[i][1]]
        directChannel = env.getChannel(i, i).getCSI()
        """signal"""
        signalPower = power * np.power(np.linalg.norm(np.matmul(directChannel, beamformer)), 4)
        """noise"""
        noisePower = dBm2num(NOISE_POWER) * np.power(np.linalg.norm(np.matmul(directChannel, beamformer)), 2)
        """interference"""
        interferencePower = 0.
        for j in range(len(actions)):
            if env.isDirectLink(i, j) or env.isIsolated(i, j):
                continue
            else:
                otherPower = dBm2num(POWER_LIST[actions[j][0]])
                otherBeamformer = BEAMFORMER_LIST[actions[j][1]]
                otherChannel = env.getChannel(j, i).getCSI()
                interferencePower += otherPower * np.power(np.linalg.norm(np.matmul(
                    np.matmul(np.matmul(beamformer.transpose().conjugate(), directChannel.transpose().conjugate()),
                              otherChannel), otherBeamformer)), 2)
        capacity.append(np.log2(1 + signalPower / (noisePower + interferencePower)))

    return capacity


def cdf(x, plot=True, *args, **kwargs):
    x, y = sorted(x), np.arange(len(x)) / len(x)
    plt.plot(x, y, *args, **kwargs) if plot else (x, y)


def action2Index(action):
    return action[0] * (POWER_LEVEL - 1) + action[1]


def index2Action(index):
    beamformer = index % CODEBOOK_SIZE
    power = index // CODEBOOK_SIZE
    return [power, beamformer]


def loadData(path=SIMULATION_DATA_PATH, name="default"):
    with open(path, 'r') as jsonFile:
        data = json.load(jsonFile)
        return data[name]


def saveData(input, path=SIMULATION_DATA_PATH, name="default"):
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    with open(path, 'w') as jsonFile:
        data[name] = input
        json.dump(data, jsonFile)


def pdf(data, *args, **kwargs):
    # create kernel, given an array it will estimate the probability over that values
    kde = gaussian_kde(data)
    # these are the values over which your kernel will be evaluated
    distSpace = np.linspace(min(data), max(data), 1000)
    # plot the results
    plt.plot(distSpace, kde(distSpace), *args, **kwargs)


def average(x):
    return sum(x) / len(x)


def mid(x):
    x = sorted(x)
    return x[len(x) // 2]


def buildCUIndexList(index):
    if index % 3 == 0:
        otherIndex1 = index + 2
        otherIndex2 = index + 1
    elif index % 3 == 1:
        otherIndex1 = index - 1
        otherIndex2 = index + 1
    else:
        otherIndex1 = index - 1
        otherIndex2 = index - 2
    return [index, otherIndex1, otherIndex2]


def sigmoid(x):
    return 1 / (1 + np.exp(-3 * x))
