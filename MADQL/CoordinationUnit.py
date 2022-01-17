import logging
import numpy as np
from Config import Config
from Sector import Sector
import random
from UserEquipment import UE
import matplotlib.pyplot as plt


class CU:
    def __init__(self, index, pos):
        """
        decisionIndex = [
            [f_0, p_0],
            [f_1, p_1],
            [f_2, p_2]
        ]
        3 * 2
        """
        np.random.seed(seed=None)
        self.index = index
        self.pos = pos  # [x, y]
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.sectors = []
        self._generateSectors_()
        self.UEs = []
        self._generateUEs_()
        # decision
        self.decisionIndex = []
        self._initDecisionIndex_()
        self.decisionIndexHistory = self.decisionIndex

    def _generateSectors_(self):
        r = self.config.cellSize
        h = self.config.BSHeight
        self.sectors.append(Sector(0, self.index, [self.pos[0] - r / 2, self.pos[1] - r / 2 * np.sqrt(3), h]))
        self.sectors.append(Sector(1, self.index, [self.pos[0] + r, self.pos[1], h]))
        self.sectors.append(Sector(2, self.index, [self.pos[0] - r / 2, self.pos[1] + r / 2 * np.sqrt(3), h]))

    def _generateUEs_(self):
        # random generate UE and fix position
        R = self.config.Rmax - self.config.Rmin
        h = self.config.UTHeight
        for i in range(3):
            r = R * np.random.rand() + self.config.Rmin
            # print("r: ", r)
            theta = (np.random.rand() * 120 + 120 * i)
            # print("theta: ", theta)
            theta = theta / 360 * 2 * np.pi
            posX = self.sectors[i].pos[0] + r * np.cos(theta)
            posY = self.sectors[i].pos[1] + r * np.sin(theta)
            # append UE in UEs
            self.UEs.append(UE(i, self.index, [posX, posY, h]))

    def getSectors(self):
        return self.sectors

    def getUEs(self):
        return self.UEs

    def _initDecisionIndex_(self):
        for i in range(3):
            self.decisionIndex.append([random.randint(0, self.config.codebookSize - 1),
                                       random.randint(0, self.config.powerLevel - 1)])

    def getDecisionIndex(self):
        return self.decisionIndex

    def getDecisionIndexHistory(self):
        return self.decisionIndexHistory

    def setDecisionIndex(self, newDecisionIndex):
        self.decisionIndexHistory = self.decisionIndex
        self.decisionIndex = newDecisionIndex

    def plotCU(self, plt=plt, local=False):
        sectorsPosX = []
        sectorsPosY = []
        UEsPosX = []
        UEsPosY = []
        cellSize = self.config.cellSize
        for i in range(3):
            sectorsPosX.append(self.sectors[i].pos[0])
            sectorsPosY.append(self.sectors[i].pos[1])
            UEsPosX.append(self.UEs[i].pos[0])
            UEsPosY.append(self.UEs[i].pos[1])
        # plot point
        plt.scatter(sectorsPosX, sectorsPosY, c='r')
        plt.scatter(UEsPosX, UEsPosY, c='b')
        plt.scatter([self.pos[0]], [self.pos[1]], c='y')
        # draw Hexagon
        theta = np.linspace(0, 2 * np.pi, 13)
        x = np.cos(theta)
        x[1::2] *= 0.5
        y = np.sin(theta)
        y[1::2] *= 0.5
        plt.plot(x[::2] * cellSize + self.pos[0], y[::2] * cellSize + self.pos[1], color='r')
        # print sector line
        point = np.linspace([-cellSize, 0], [0, 0], 10) + self.pos
        plt.plot(point[:, 0], point[:, 1], color='k')
        point = np.linspace([0, 0], [cellSize / 2, cellSize / 2 * np.sqrt(3)], 10) + self.pos
        plt.plot(point[:, 0], point[:, 1], color='k')
        point = np.linspace([0, 0], [cellSize / 2, - cellSize / 2 * np.sqrt(3)], 10) + self.pos
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
    cu.plotCU(local=True)
    # plot_Hexagon()
