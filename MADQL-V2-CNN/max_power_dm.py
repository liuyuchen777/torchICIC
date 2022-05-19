import logging
import random

from config import *
from utils import Algorithm


class MaxPower:
    def __init__(self):
        self.logger = logging.getLogger()
        self.algorithm = Algorithm.MAX_POWER
        self.linkNumber = 3 * CELL_NUMBER

    def takeAction(self):
        actions = []
        for _ in range(self.linkNumber):
            actions.append([POWER_LEVEL - 1, random.randint(0, CODEBOOK_SIZE - 1)])
        return actions
