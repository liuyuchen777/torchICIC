from sector import Sector
from user_equipment import UE
from config import Config
import numpy as np


def generateSector(index, pos):
    config = Config()
    sectors = []
    r = config.cellSize
    h = config.BSHeight
    sectors.append(Sector(0, index, [pos[0] - r / 2, pos[1] - r / 2 * np.sqrt(3), h]))
    sectors.append(Sector(1, index, [pos[0] + r, pos[1], h]))
    sectors.append(Sector(2, index, [pos[0] - r / 2, pos[1] + r / 2 * np.sqrt(3), h]))

    return sectors


def generateUE(index, sectors):
    # random generate UE and fix position
    config = Config()
    UEs = []
    R = config.Rmax - config.Rmin
    h = config.UTHeight
    for i in range(3):
        # generate r and theta
        r = R * np.random.rand() + config.Rmin
        theta = (np.random.rand() * 120 + 120 * i)
        theta = theta / 360 * 2 * np.pi
        # r-theta to x-y
        posX = sectors[i].pos[0] + r * np.cos(theta)
        posY = sectors[i].pos[1] + r * np.sin(theta)
        # append UE in UEs
        UEs.append(UE(i, index, [posX, posY, h]))

    return UEs
