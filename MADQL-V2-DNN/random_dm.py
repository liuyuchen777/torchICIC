import logging
import random

from config import *
from utils import Algorithm


def takeActionRandom(linkNumber):
    actions = []
    for _ in range(linkNumber):
        actions.append([random.randint(0, POWER_LEVEL - 1), random.randint(0, CODEBOOK_SIZE - 1)])
    return actions


class Random:
    def __init__(self):
        self.logger = logging.getLogger()
        self.algorithm = Algorithm.RANDOM
        self.linkNumber = 3 * CELL_NUMBER

    def takeAction(self):
        return takeActionRandom(self.linkNumber)
