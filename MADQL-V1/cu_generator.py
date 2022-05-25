import numpy as np
from coordination_unit import CU
from config import *


def generateCU():
    """
    CU position generator

    Returns:
        CUs: list of CU
    """
    CUs = []
    if CELL_NUMBER < 1 or CELL_NUMBER > 7:
        raise Exception("Incorrect setting of cell number: " + CELL_NUMBER)
    # max support 7 cells
    CUNumber = CELL_NUMBER
    cellRadius = CELL_SIZE * np.sqrt(3)
    # CU Index 0
    CUs.append(CU(0, [0., 0.]))
    for i in range(1, CUNumber):
        theta = np.pi / 6 + (i - 1) * np.pi / 3
        posX = cellRadius * np.cos(theta)
        posY = cellRadius * np.sin(theta)
        CUs.append(CU(i, [posX, posY]))

    return CUs
