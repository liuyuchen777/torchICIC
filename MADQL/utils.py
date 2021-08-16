import logging
import sys
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


def set_logger():
    config = Config()
    log_file_path = "./log/" + time.strftime("%Y-%m-%d") + ".log"
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S', level=logging.DEBUG,
                        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()])
    # initial log
    logging.info("----------------------------CONFIG--------------------------------------------")
    logging.info(f'DATE: {time.strftime("%Y-%m-%d", time.localtime())}, '
                 f'TIME: {time.strftime("%H:%M:%S", time.localtime())}')
    logging.info(f'Power Level: {config.power_level}, Codebook Size: {config.codebook_size}, '
                 f'Cell Length: {config.cell_length}')
    # network config information
    logging.info("-----------------------------END----------------------------------------------")


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
    MADQL = 2
    FP = 3
    MAX_POWER = 4


if __name__ == "__main__":
    # test Logger
    set_logger()
    logger = logging.getLogger(__name__)
    logger.debug("Hello World")
