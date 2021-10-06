import logging
import numpy as np
from config import Config
from sector import Sector
import time
import random
from user_equipment import UE
import matplotlib.pyplot as plt


class CU:
    def __init__(self, index, pos):
        np.random.seed(seed=None)
        self.index = index
        self.pos = pos  # [x, y]
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.sectors = []
        self._generate_sectors_()
        self.UEs = []
        self._generate_UEs_()
        # decision
        self.decision_index = []
        self._init_decision_index()
        self.decision_index_history = self.decision_index

    def _generate_sectors_(self):
        r = self.config.cell_length
        h = self.config.BS_height
        self.sectors.append(Sector(0, self.index, [self.pos[0] - r / 2, self.pos[1] - r / 2 * np.sqrt(3), h]))
        self.sectors.append(Sector(1, self.index, [self.pos[0] + r, self.pos[1], h]))
        self.sectors.append(Sector(2, self.index, [self.pos[0] - r / 2, self.pos[1] + r / 2 * np.sqrt(3), h]))

    def _generate_UEs_(self):
        # random generate UE and fix position
        R = self.config.R_max - self.config.R_min
        h = self.config.UT_height
        for i in range(3):
            r = R * np.random.rand() + self.config.R_min
            # print("r: ", r)
            theta = (np.random.rand() * 120 + 120 * i)
            # print("theta: ", theta)
            theta = theta / 360 * 2 * np.pi
            pos_x = self.sectors[i].pos[0] + r * np.cos(theta)
            pos_y = self.sectors[i].pos[1] + r * np.sin(theta)
            # append UE in UEs
            self.UEs.append(UE(i, self.index, [pos_x, pos_y, h]))

    def get_sectors(self):
        return self.sectors

    def get_UEs(self):
        return self.UEs

    def _init_decision_index(self):
        for i in range(3):
            self.decision_index.append([random.randint(0, self.config.codebook_size-1),
                                        random.randint(0, self.config.power_level-1)])

    def get_decision_index(self):
        return self.decision_index

    def get_decision_index_history(self):
        return self.decision_index_history

    def set_decision_index(self, new_decision_index):
        self.decision_index_history = self.decision_index
        self.decision_index = new_decision_index

    def plot_CU(self, plt=plt, local=False):
        sectors_pos_x = []
        sectors_pos_y = []
        UEs_pos_x = []
        UEs_pos_y = []
        for i in range(3):
            sectors_pos_x.append(self.sectors[i].pos[0])
            sectors_pos_y.append(self.sectors[i].pos[1])
            UEs_pos_x.append(self.UEs[i].pos[0])
            UEs_pos_y.append(self.UEs[i].pos[1])
        # plot point
        plt.scatter(sectors_pos_x, sectors_pos_y, c='r')
        plt.scatter(UEs_pos_x, UEs_pos_y, c='b')
        plt.scatter([self.pos[0]], [self.pos[1]], c='y')
        # draw Hexagon
        theta = np.linspace(0, 2 * np.pi, 13)
        x = np.cos(theta)
        x[1::2] *= 0.5
        y = np.sin(theta)
        y[1::2] *= 0.5
        plt.plot(x[::2] * 200. + self.pos[0], y[::2] * 200. + self.pos[1], color='r')
        # print sector line
        point = np.linspace([-200, 0], [0, 0], 10) + self.pos
        plt.plot(point[:, 0], point[:, 1], color='k')
        point = np.linspace([0, 0], [100, 100*np.sqrt(3)], 10) + self.pos
        plt.plot(point[:, 0], point[:, 1], color='k')
        point = np.linspace([0, 0], [100, -100 * np.sqrt(3)], 10) + self.pos
        plt.plot(point[:, 0], point[:, 1], color='k')
        # show figure
        if local:
            plt.show()


def plot_Hexagon():
    theta = np.linspace(0, 2 * np.pi, 13)
    x = np.cos(theta)
    x[1::2] *= 0.5
    y = np.sin(theta)
    y[1::2] *= 0.5
    plt.plot(x[::2]*200., y[::2]*200., color='r')
    plt.show()


if __name__ == "__main__":
    cu = CU(0, [0., 0.])
    cu.plot_CU(local=True)
    # plot_Hexagon()
