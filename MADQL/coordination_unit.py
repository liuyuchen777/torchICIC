import logging
import numpy as np
from config import Config
from sector import Sector
import time
import random
from user_equipment import UE


class CU:
    def __init__(self, index, pos):
        self.index = index
        self.pos = pos  # [x, y]
        self.config = Config()
        self.logger = logging.getLogger(__name__)
        self.sectors = []
        self._generate_sectors_()
        self.UEs = []
        self._generate_UEs_()
        self.seed = time.time()
        random.seed(self.seed)
        # decision
        self.decision_index = []
        self._init_decision_index()
        self.decision_index_history = self.decision_index

    def _generate_sectors_(self):
        r = self.config.cell_length
        h = self.config.BS_height
        self.sectors.append(Sector(0, self.index, [self.pos[0] - r / 2, self.pos[1] + r / 2 * np.sqrt(3), h]))
        self.sectors.append(Sector(1, self.index, [self.pos[0] - r / 2, self.pos[1] - r / 2 * np.sqrt(3), h]))
        self.sectors.append(Sector(2, self.index, [self.pos[0] + r, self.pos[1], h]))

    def _generate_UEs_(self):
        # random generate UE and fix position
        R = self.config.cell_length / 2 * np.sqrt(3)
        h = self.config.UT_height
        for i in range(3):
            r = R * random.random()
            theta = (random.randint((120 * i + 240) % 360, (120 * i + 359) % 360)) / 360 * 2 * np.pi
            pos_x = self.sectors[i].pos[0] + r * np.cos(theta)
            pos_y = self.sectors[i].pos[1] + r * np.sin(theta)
            # append UE in UEs
            self.UEs.append(UE(i, self.index, [pos_x, pos_y, h]))

    def get_sectors(self):
        return self.sectors

    def get_UEs(self):
        return self.UEs

    def _init_decision_index(self):
        max_index = self.config.codebook_size * self.config.power_level
        for i in range(3):
            self.decision_index.append(random.randint(0, max_index))

    def get_decision_index(self):
        return self.decision_index

    def get_decision_index_history(self):
        return self.decision_index_history

    def set_decision_index(self, new_decision_index):
        self.decision_index_history = self.decision_index
        self.decision_index = new_decision_index


if __name__ == "__main__":
    cu = CU(0, [0., 0.], Logger(debug_tag=True))

