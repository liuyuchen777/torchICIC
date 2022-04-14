import numpy as np
from coordination_unit import CU
from config import Config


def generateCU():
    """
    CU position generator

    Returns:
        CUs: list of CU
    """
    config = Config()
    CUs = []
    if config.cellNumber < 1 or config.cellNumber > 7:
        raise Exception("Incorrect setting of cell number: " + config.cellNumber)
    # max support 7 cells
    CUNumber = config.cellNumber
    cellRadius = config.cellSize * np.sqrt(3)
    # CU Index 0
    CUs.append(CU(0, [0., 0.]))
    for i in range(1, CUNumber):
        theta = np.pi / 6 + (i - 1) * np.pi / 3
        posX = cellRadius * np.cos(theta)
        posY = cellRadius * np.sin(theta)
        CUs.append(CU(i, [posX, posY]))

    return CUs
