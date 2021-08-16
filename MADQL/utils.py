import logging
from enum import Enum

import numpy as np
import time
from config import Config


def dB2num(dB):
    num = 10 ** (dB / 10)
    return num


def num2dB(num):
    dB = 10 * np.log10(num)
    return dB


def decode_index(index):
    """
    compose of CU decision index: [S1 decision index, S2 decision index, S3 decision index]
    decision index = power level used * 10 + beamformer used
    """
    power_level = index / 10
    beamformer_index = index % 10
    return [power_level, beamformer_index]


class Logger:
    def __init__(self, log_path="./log/", debug_tag=False):
        # set path
        self.config = Config()
        self.log_file_path = log_path + time.strftime("%Y-%m-%d") + ".log"
        logging.basicConfig(filename=self.log_file_path, format='%(message)s', level=logging.DEBUG)
        self.debug = debug_tag
        self._log_start_()

    def _log_start_(self):
        # starting write some information
        self.log("----------------------------CONFIG----------------------------------")
        self.log(f'DATE: {time.strftime("%Y-%m-%d", time.localtime())}, '
                 f'TIME: {time.strftime("%H:%M:%S", time.localtime())}')
        self.log(f'Power Level: {self.config.power_level}, Codebook Size: {self.config.codebook_size}, '
                 f'Cell Length: {self.config.cell_length}')
        # network config information
        self.log("-----------------------------END------------------------------------")

    def log(self, log_item):
        print(log_item)
        logging.info(log_item)

    def log_l(self, log_item):
        print("[Learning]", log_item)
        logging.info("[Learning] " + log_item)

    def log_c(self, log_item):
        print("[Communication]", log_item)
        logging.info("[Communication] " + log_item)

    def log_d(self, log_item):
        if self.debug:
            print("[DEBUG]", f'[{__file__}]', log_item)


class Algorithm(Enum):
    RANDOM = 1
    MADQL = 2
    FP = 3
    MAX_POWER = 4


if __name__ == "__main__":
    # test Logger
    log = Logger(debug_tag=True)
    log.log_l("Hello World!")
    log.log("Hello World!")
    log.log_d("Hello World!")