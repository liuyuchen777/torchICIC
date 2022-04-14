import logging
import numpy as np
from config import Config
import random
import matplotlib.pyplot as plt
from bs_ue_generator import generateUE, generateSector


class CU:

    def __init__(self, index, pos):
        """
        action = [
            [f_0, p_0],
            [f_1, p_1],
            [f_2, p_2]
        ]
        3 * 2
        """
        self.index = index
        self.pos = pos  # [x, y]
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.sectors = generateSector(self.index, self.pos)
        self.UEs = generateUE(self.index, self.sectors)
        # decision
        self.action = []
        self._initAction_()
        self.actionHistory = self.action

    def getSectors(self):
        return self.sectors

    def getUEs(self):
        return self.UEs

    def _initAction_(self):
        for i in range(3):
            self.action.append([random.randint(0, self.config.codebookSize - 1),
                                random.randint(0, self.config.powerLevel - 1)])

    def getAction(self):
        return self.action

    def getActionHistory(self):
        return self.actionHistory

    def setAction(self, newAction):
        self.actionHistory = self.action
        self.action = newAction


"""
Single CU plot util
"""


def plotCU(CU, plt=plt, local=False):
    sectorsPosX = []
    sectorsPosY = []
    UEsPosX = []
    UEsPosY = []
    cellSize = CU.config.cellSize
    for i in range(3):
        sectorsPosX.append(CU.sectors[i].pos[0])
        sectorsPosY.append(CU.sectors[i].pos[1])
        UEsPosX.append(CU.UEs[i].pos[0])
        UEsPosY.append(CU.UEs[i].pos[1])
    # plot point
    plt.scatter(sectorsPosX, sectorsPosY, c='r')
    plt.scatter(UEsPosX, UEsPosY, c='b')
    plt.scatter([CU.pos[0]], [CU.pos[1]], c='y')
    # draw Hexagon
    theta = np.linspace(0, 2 * np.pi, 13)
    x = np.cos(theta)
    x[1::2] *= 0.5
    y = np.sin(theta)
    y[1::2] *= 0.5
    plt.plot(x[::2] * cellSize + CU.pos[0], y[::2] * cellSize + CU.pos[1], color='r')
    # print sector line
    point = np.linspace([-cellSize, 0], [0, 0], 10) + CU.pos
    plt.plot(point[:, 0], point[:, 1], color='k')
    point = np.linspace([0, 0], [cellSize / 2, cellSize / 2 * np.sqrt(3)], 10) + CU.pos
    plt.plot(point[:, 0], point[:, 1], color='k')
    point = np.linspace([0, 0], [cellSize / 2, - cellSize / 2 * np.sqrt(3)], 10) + CU.pos
    plt.plot(point[:, 0], point[:, 1], color='k')
    # show figure
    if local:
        plt.show()


def plotHexagon():
    theta = np.linspace(0, 2 * np.pi, 13)
    x = np.cos(theta)
    x[1::2] *= 0.5
    y = np.sin(theta)
    y[1::2] *= 0.5
    plt.plot(x[::2]*200., y[::2]*200., color='r')
    plt.show()


if __name__ == "__main__":
    cu = CU(0, [0., 0.])
    plotCU(cu, local=True)
    # plot_Hexagon()
