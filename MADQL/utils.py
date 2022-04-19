import logging
from enum import Enum
import numpy as np
from config import *
import time
from decision_maker import Random, MADQL, MaxPower, WMMSE, FP, CellES
import matplotlib.pyplot as plt

"""
Functions to convert number to dBm/dB (power)
"""


def dBm2num(dB):
    num = np.power(10., dB / 10.) / 1000.
    return num


def dB2num(dB):
    num = np.power(10., dB / 10.)
    return num


def num2dB(num):
    dB = 10. * np.log10(num)
    return dB


"""
Channel index convertor
Downlink transmission
index = [transmitCU.index, transmitSector.index, receiverCU.index, receiverSector.index]
"""


def index2str(index):
    return f'{index[0]},{index[1]},{index[2]},{index[3]}'


def str2index(index_str):
    index = []
    index_list = index_str.split(',')
    for s in index_list:
        index.append(int(s))
    return index


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
    logging.info(f'Batch Size: {BATCH_SIZE}, Total time slot: {TOTAL_TIME_SLOT}, '
                 f'Learning Rate: {LEARNING_RATE}, Reg Beta: {REG_BETA}')
    logging.info(f'Gamma: {GAMMA}, Epsilon: {EPSILON}')
    # network config information
    logging.info("=========================================END=========================================")


"""
Enum class of current available method
"""


class Algorithm(Enum):
    RANDOM = 1
    MAX_POWER = 2
    FP = 3
    WMMSE = 4
    MADQL = 5
    CELL_ES = 6


def setDecisionMaker(algorithm):
    """
    Return instance of decision maker base on algorithm
    Args:
        algorithm:

    Returns:
        decision maker
    """
    if algorithm == Algorithm.RANDOM:
        return Random()
    elif algorithm == Algorithm.MAX_POWER:
        return MaxPower()
    elif algorithm == Algorithm.FP:
        return FP()
    elif algorithm == Algorithm.WMMSE:
        return WMMSE()
    elif algorithm == Algorithm.MADQL:
        return MADQL()
    elif algorithm == Algorithm.CELL_ES:
        return CellES()
    else:
        raise Exception("Incorrect algorithm setting: " + algorithm)


"""
Inter-Sector of a base station will isolate with each other
CU index start from 0, anti-clockwise from -30 degree
Sector index start from 0, anti-clockwise from -120 degree
"""


neighborTable = [
    [1, 2, 3, 4, 5, 6],
    [0, 2, 6],
    [0, 1, 3],
    [0, 2, 4],
    [0, 3, 5],
    [0, 4, 6],
    [0, 1, 5]
]


skipTable = [
    [-1, 0, 0, 1, 1, 2, 2],     # 0
    [1, -1, 1, 1, -1, -1, 2],     # 1
    [2, 2, -1, 1, -1, -1, -1],     # 2
    [2, -1, 0, -1, 2, -1, -1],     # 3
    [0, -1, -1, 0, -1, 2, -1],     # 4
    [0, -1, -1, -1, 1, -1, 0],     # 5
    [1, 0, -1, -1, -1, 1, -1]      # 6
]


def judgeSkip(index):
    """
    index is sequence
    [receiver, sender]
    """
    if skipTable[index[2]][index[0]] == index[1]:
        return True
    else:
        return False


def cdf(x, plot=True, *args, **kwargs):
    x, y = sorted(x), np.arange(len(x)) / len(x)
    plt.plot(x, y, *args, **kwargs) if plot else (x, y)


if __name__ == "__main__":
    # test Logger
    setLogger()
    logger = logging.getLogger(__name__)
    logger.debug("Hello World")
    logger.info("Hello World")
    # str2index, index2str
    print(str2index("1,2,3,4"))
