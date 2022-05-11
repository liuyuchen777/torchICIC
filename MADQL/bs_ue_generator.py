from sector import Sector
from user_equipment import UE
import numpy as np
from config import *

"""
When use single model for all CU, we need to keep position of UE same in each CU
"""

SAME_DISTRIBUTION = False

radius_random_seed = [np.random.rand(), np.random.rand(), np.random.rand()]
angle_random_seed = [np.random.rand(), np.random.rand(), np.random.rand()]


def generateSector(index, pos):
    sectors = []
    r = CELL_SIZE
    h = BS_HEIGHT
    sectors.append(Sector(0, index, [pos[0] - r / 2, pos[1] - r / 2 * np.sqrt(3), h]))
    sectors.append(Sector(1, index, [pos[0] + r, pos[1], h]))
    sectors.append(Sector(2, index, [pos[0] - r / 2, pos[1] + r / 2 * np.sqrt(3), h]))

    return sectors


def generateUE(index, sectors):
    """random generate UE and fix position"""
    UEs = []
    R = R_MAX - R_MIN
    h = UT_HEIGHT
    for i in range(3):
        # generate r and theta
        if SAME_DISTRIBUTION:
            r = R * radius_random_seed[i] + R_MIN
            theta = (angle_random_seed[i] * 120 + 120 * i)
        else:
            r = R * np.random.rand() + R_MIN
            theta = (np.random.rand() * 120 + 120 * i)
        theta = theta / 360 * 2 * np.pi
        # r-theta to x-y
        posX = sectors[i].pos[0] + r * np.cos(theta)
        posY = sectors[i].pos[1] + r * np.sin(theta)
        # append UE in UEs
        UEs.append(UE(i, index, [posX, posY, h]))

    return UEs
