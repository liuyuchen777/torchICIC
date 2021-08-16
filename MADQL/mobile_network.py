from config import Config
from utils import Algorithm, set_logger
import matplotlib.pyplot as plt
import numpy as np
from coordination_unit import CU
from memory_pool import MemoryPool
from decision_maker import DecisionMaker
from environment import Environment
import logging


class MobileNetwork:
    def __init__(self, algorithm=Algorithm.RANDOM):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.CUs = []
        self._generate_CUs_()
        self.mp = MemoryPool()
        self.dm = DecisionMaker(algorithm)
        # generate environment
        self.env = Environment(self.CUs)

    def _generate_CUs_(self):
        # max support 7 cells
        CU_number = self.config.cell_number
        R = self.config.cell_length * np.sqrt(3)
        # the first CU
        self.CUs.append(CU(0, [0., 0.]))
        for i in range(1, CU_number):
            theta = np.pi/6 + (i-1)*np.pi/3
            pos_x = R * np.cos(theta)
            pos_y = R * np.sin(theta)
            self.CUs.append(CU(i, [pos_x, pos_y]))

    def print_CU_position(self):
        for i in range(len(self.CUs)):
            logging.info(f'Position of {i}th CU is: {self.CUs[i].pos}')

    def plot_CU_position(self):
        logging.info("---------------Plot CUs--------------")
        CU_pos_x = []
        CU_pos_y = []
        for CU in self.CUs:
            CU_pos_x.append(CU.pos[0])
            CU_pos_y.append(CU.pos[1])
        fig, ax = plt.subplots()
        plt.scatter(CU_pos_x, CU_pos_y, c='r')
        # set axis
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data', 0))
        ax.yaxis.set_ticks_position('left')
        ax.spines['left'].set_position(('data', 0))
        # show figure
        plt.show()


if __name__ == "__main__":
    set_logger()
    mn = MobileNetwork()
    # mn.print_CU_position()
    # mn.plot_CU_position()
    # mn.env.print_channel()
