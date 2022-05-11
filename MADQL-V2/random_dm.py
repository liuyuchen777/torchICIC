import logging
import random

from config import *
from utils import Algorithm


class Random:
    def __init__(self):
        self.logger = logging.getLogger()
        self.algorithm = Algorithm.RANDOM

    def takeAction(self):
        return [random.randint(0, POWER_LEVEL - 1), random.randint(0, CODEBOOK_SIZE - 1)]
