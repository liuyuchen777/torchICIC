import logging
import random

from config import *
from utils import Algorithm


class MaxPower:
    def __init__(self):
        self.logger = logging.getLogger()
        self.algorithm = Algorithm.MAX_POWER

    def takeAction(self):
        return [POWER_LEVEL - 1, random.randint(0, CODEBOOK_SIZE - 1)]
