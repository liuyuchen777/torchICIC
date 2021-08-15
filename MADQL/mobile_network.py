from config import Config
from utils import Logger, Algorithm
import matplotlib.pyplot as plt
import numpy as np
from coordination_unit import CU
from memory_pool import MemoryPool
from DecisionMaker import DecisionMaker


class MobileNetwork:
    def __init__(self, algorithm=Algorithm.RANDOM):
        self.logger = Logger(debug_tag=True)
        self.config = Config()
        self.CUs = []
        self._generate_CUs_()
        self.mp = MemoryPool()
        self.dm = DecisionMaker(algorithm)

    def _generate_CUs_(self):
        # max support 7 cells
        CU_number = self.config.cell_number
        R = self.config.cell_length * np.sqrt(3)
        self.logger.log_d(f"number of CU: {CU_number}")
        # the first CU
        self.CUs.append(CU(0, [0., 0.], self.logger, self.config))
        for i in range(1, CU_number):
            theta = np.pi/6 + (i-1)*np.pi/3
            pos_x = R * np.cos(theta)
            pos_y = R * np.sin(theta)
            self.CUs.append(CU(i, [pos_x, pos_y], self.logger, self.config))

    def print_CU_position(self):
        for i in range(len(self.CUs)):
            print(f'Position of {i}th CU is: {self.CUs[i].pos}')

    def plot_CU_position(self):
        print("---------------Plot CUs--------------")
        CU_pos_x = []
        CU_pos_y = []
        for CU in self.CUs:
            CU_pos_x.append(CU.pos[0])
            CU_pos_y.append(CU.pos[1])
        fig, ax = plt.subplots()
        plt.scatter(CU_pos_x, CU_pos_y, c='r')

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data', 0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data', 0))

        plt.show()


if __name__ == "__main__":
    mn = MobileNetwork()
    mn.print_CU_position()
    mn.plot_CU_position()
