import numpy as np
from config import Config
from sector import Sector
import time
import random
from utils import Logger


class CU:
    def __init__(self, index, pos, logger, config=Config()):
        self.index = index
        self.pos = pos  # [x, y]
        self.config = config
        self.logger = logger
        self.sectors = []
        self._generate_sectors_()
        self.UEs = []
        self._generate_UEs_()
        self.seed = time.time()
        random.seed(self.seed)
        # decision


    def _generate_sectors_(self):
        r = self.config.cell_length
        h = self.config.BS_height
        self.sectors.append(Sector(0, self.index, [self.pos[0] - r / 2, self.pos[1] + r / 2 * np.sqrt(3), h],
                                   self.logger, self.config))
        self.sectors.append(Sector(1, self.index, [self.pos[0] - r / 2, self.pos[1] - r / 2 * np.sqrt(3), h],
                                   self.logger, self.config))
        self.sectors.append(Sector(2, self.index, [self.pos[0] + r, self.pos[1], h],
                                   self.logger, self.config))
        for sector in self.sectors:
            self.logger.log_d(f'sector {sector.index}\'s position is {sector.pos}')

    def _generate_UEs_(self):
        # random generate UE and fix position
        R = self.config.cell_length / 2 * np.sqrt(3)
        for i in range(3):
            r = R * random.random()
            theta = (random.randint((120 * i + 240) % 360, (120 * i + 359) % 360))
            self.logger.log_d(f'r = {r}, theta = {theta}')
            # append UE in UEs

        for UE in self.UEs:
            self.logger.log_d(f'UE {UE.index}\'s position is {UE.pos}')

    def get_sectors(self):
        return self.sectors

    def get_UEs(self):
        return self.UEs


if __name__ == "__main__":
    cu = CU(0, [0., 0.], Logger(debug_tag=False))

