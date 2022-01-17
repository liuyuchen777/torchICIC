import logging
from enum import Enum
import numpy as np
from Config import Config
import time


def dBm2num(dB):
    num = np.power(10., dB / 10.) / 1000.
    return num


def dB2num(dB):
    num = np.power(10., dB / 10.)
    return num


def num2dB(num):
    dB = 10. * np.log10(num)
    return dB


def index2str(index):
    return f'{index[0]},{index[1]},{index[2]},{index[3]}'


def str2index(index_str):
    index = []
    index_list = index_str.split(',')
    for s in index_list:
        index.append(int(s))
    return index


def setLogger(file=True, debug=False):
    config = Config()
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
    logging.info(f'Power Level: {config.powerLevel}, Codebook Size: {config.codebookSize}, '
                 f'Cell Length: {config.cellSize} m, Cell Number: {config.cellNumber}')
    logging.info(f'Path Loss Exponent: {config.alpha}, Log-normal Sigma: {config.ShadowingSigma} db, '
                 f'Gaussian Sigma: {config.gaussianSigma} db')
    logging.info("-----------------------------------------DL------------------------------------------")
    logging.info(f'Batch Size: {config.batchSize}, Total time slot: {config.totalTimeSlot}, '
                 f'Learning Rate: {config.learningRate}, Reg Beta: {config.regBeta}')
    logging.info(f'Gamma: {config.gamma}, Epsilon: {config.epsilon}')
    # network config information
    logging.info("=========================================END=========================================")


class Algorithm(Enum):
    RANDOM = 1
    MAX_POWER = 2
    FP = 3
    WMMSE = 4
    MADQL = 5
    CELL_ES = 6


neighborTable = [
    [1, 2, 3, 4, 5, 6],
    [0, 2, 6],
    [0, 1, 3],
    [0, 2, 4],
    [0, 3, 5],
    [0, 4, 6],
    [0, 1, 5]
]

# NOTE: CU中每个sector受到影响的Inter-Cell的sector id是一样的，skipTable中记录对应CU要跳过的sector id
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
    # index = [cu.index, sector.index, otherCU.index, otherCUSector.index]
    if skipTable[index[2]][index[0]] == index[1]:
        return True
    else:
        return False


if __name__ == "__main__":
    # test Logger
    setLogger()
    logger = logging.getLogger(__name__)
    logger.debug("Hello World")
    logger.info("Hello World")
    # str2index, index2str
    print(str2index("1,2,3,4"))
