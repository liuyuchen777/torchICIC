import logging

from utils import Algorithm, calCapacity
from config import *


class CellES:
    """Only Can be Used When CELL_NUMBER = 1"""
    def __init__(self):
        self.logger = logging.getLogger()
        self.algorithm = Algorithm.CELL_ES

    def takeAction(self, channels):
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
                                tmpCapacity = sum(calCapacity(tmpActions, channels))
                                if tmpCapacity > maxCapacity:
                                    maxCapacity = tmpCapacity
                                    actions = tmpActions
        return actions
