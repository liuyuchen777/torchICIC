import logging
import random

from utils import Algorithm, calCapacity
from config import *


def beamCellES(env):
    actions = []
    maxCapacity = 0.
    for beamformer1 in range(CODEBOOK_SIZE):
        for beamformer2 in range(CODEBOOK_SIZE):
            for beamformer3 in range(CODEBOOK_SIZE):
                tmpActions = [
                    [random.randint(0, POWER_LEVEL - 1), beamformer1],
                    [random.randint(0, POWER_LEVEL - 1), beamformer2],
                    [random.randint(0, POWER_LEVEL - 1), beamformer3]
                ]
                tmpCapacity = sum(calCapacity(tmpActions, env))
                if tmpCapacity > maxCapacity:
                    maxCapacity = tmpCapacity
                    actions = tmpActions
    return actions


def powerCellES(env):
    actions = []
    maxCapacity = 0.
    for power1 in range(POWER_LEVEL):
        for power2 in range(POWER_LEVEL):
            for power3 in range(POWER_LEVEL):
                for beamformer3 in range(CODEBOOK_SIZE):
                    tmpActions = [
                        [power1, random.randint(0, CODEBOOK_SIZE - 1)],
                        [power2, random.randint(0, CODEBOOK_SIZE - 1)],
                        [power3, random.randint(0, CODEBOOK_SIZE - 1)]
                    ]
                    tmpCapacity = sum(calCapacity(tmpActions, env))
                    if tmpCapacity > maxCapacity:
                        maxCapacity = tmpCapacity
                        actions = tmpActions
    return actions


def powerBeamCellES(env):
    actions = []
    maxCapacity = 0.
    for power1 in range(POWER_LEVEL):
        for beamformer1 in range(CODEBOOK_SIZE):
            for power2 in range(POWER_LEVEL):
                for beamformer2 in range(CODEBOOK_SIZE):
                    for power3 in range(POWER_LEVEL):
                        for beamformer3 in range(CODEBOOK_SIZE):
                            tmpActions = [
                                [power1, beamformer1],
                                [power2, beamformer2],
                                [power3, beamformer3]
                            ]
                            tmpCapacity = sum(calCapacity(tmpActions, env))
                            if tmpCapacity > maxCapacity:
                                maxCapacity = tmpCapacity
                                actions = tmpActions
    return actions


class CellES:
    """Only Can be Used When CELL_NUMBER = 1"""

    def __init__(self):
        self.logger = logging.getLogger()
        self.algorithm = Algorithm.CELL_ES

    def takeAction(self, env):
        return powerCellES(env)
