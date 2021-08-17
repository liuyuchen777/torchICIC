import logging
from enum import Enum
import numpy as np
from config import Config
import time


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


def set_logger(file=True, debug=True):
    config = Config()
    if debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    if file:
        log_file_path = "./log/" + time.strftime("%Y-%m-%d") + ".log"
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%H:%M:%S', level=log_level,
                            handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
    else:
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                            datefmt='%H:%M:%S', level=log_level)
    # initial log
    logging.info("=====================================CONFIG=========================================")
    logging.info(f'START TIME: {time.strftime("%H:%M:%S", time.localtime())}')
    logging.info("-----------------------------------COMMUNICATION------------------------------------")
    logging.info(f'Power Level: {config.power_level}, Codebook Size: {config.codebook_size}, '
                 f'Cell Length: {config.cell_length} m, Cell Number: {config.cell_number}')
    logging.info(f'Path Loss Exponent: {config.alpha}, Lognormal Sigma: {config.log_normal_sigma} db, '
                 f'Lambda: {config.wave_length} m, Gaussian Sigma: {config.Gaussian_sigma} db')
    logging.info("-----------------------------------------DL------------------------------------------")
    # network config information
    logging.info("=========================================END=========================================")


def decode_index(index):
    """
    compose of CU decision index: [S1 decision index, S2 decision index, S3 decision index]
    decision index = power level used * 10 + beamformer used
    """
    power_level = index / 10
    beamformer_index = index % 10
    return [power_level, beamformer_index]


class Algorithm(Enum):
    RANDOM = 1
    MAX_POWER = 2
    FP = 3
    MADQL = 4


neighbor_table = [
    [1, 2, 3, 4, 5, 6],
    [0, 2, 6],
    [0, 1, 3],
    [0, 2, 4],
    [0, 3, 5],
    [0, 4, 6],
    [0, 1, 5]
]


def judge_skip(index):
    print("Under Construct")

    return False


if __name__ == "__main__":
    # test Logger
    set_logger()
    logger = logging.getLogger(__name__)
    logger.debug("Hello World")
    logger.info("Hello World")
    # str2index, index2str
    print(str2index("1,2,3,4"))
