import logging
from enum import Enum
import numpy as np
from Config import Config
import time


def dBm2num(dB):
    num = 10 ** (dB / 10) / 1000
    return num


def dB2num(dB):
    num = 10 ** (dB / 10)
    return num


def num2dB(num):
    dB = 10 * np.log10(num)
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
                 f'Cell Length: {config.cellLength} m, Cell Number: {config.cellNumber}')
    logging.info(f'Path Loss Exponent: {config.alpha}, Lognormal Sigma: {config.logNormalSigma} db, '
                 f'Gaussian Sigma: {config.gaussianSigma} db')
    logging.info("-----------------------------------------DL------------------------------------------")
    # network config information
    logging.info("=========================================END=========================================")


def decodeIndex(index):
    """
    compose of CU decision index: [S1 decision index, S2 decision index, S3 decision index]
    decision index = power level used * 10 + beamformer used
    """
    powerLevel = index / 10
    beamformerIndex = index % 10
    return [powerLevel, beamformerIndex]


class Algorithm(Enum):
    RANDOM = 1
    MAXPOWER = 2
    FP = 3
    WMMSE = 4
    MADQL = 5


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
    [0, 0, 1, 1, 2, 2],
    [],
    [],
    [],
    [],
    []
]


def judgeSkip(index):
    """index is sequence"""
    # index = [cu.index, sector.index, otherCU.index, otherCUSector.index]


    return False


if __name__ == "__main__":
    # test Logger
    setLogger()
    logger = logging.getLogger(__name__)
    logger.debug("Hello World")
    logger.info("Hello World")
    # str2index, index2str
    print(str2index("1,2,3,4"))
