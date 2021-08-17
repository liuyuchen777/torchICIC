from config import Config
from utils import Algorithm, set_logger
import matplotlib.pyplot as plt
import numpy as np
from coordination_unit import CU
from memory_pool import MemoryPool
from decision_maker import MaxPower, MADQL, Random
from environment import Environment
import logging


def set_decision_maker(algorithm):
    if algorithm == Algorithm.RANDOM:
        return Random()
    elif algorithm == Algorithm.MAX_POWER:
        return MaxPower()
    elif algorithm == Algorithm.MADQL:
        return MADQL()


class MobileNetwork:
    def __init__(self, algorithm=Algorithm.RANDOM):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.CUs = []
        self._generate_CUs_()
        self.mp = MemoryPool()
        self.algorithm = algorithm
        self.dm = set_decision_maker(algorithm)
        # generate environment
        self.env = Environment(self.CUs)
        self.reward_record = []

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
        CU_pos_x = []
        CU_pos_y = []
        for CU in self.CUs:
            CU_pos_x.append(CU.pos[0])
            CU_pos_y.append(CU.pos[1])
        plt.scatter(CU_pos_x, CU_pos_y, c='r')
        # show figure
        plt.show()

    def plot_mobile_network(self):
        for cu in self.CUs:
            cu.plot_CU(plt=plt)
        plt.show()

    def build_state(self, cu_index):
        print("Under Construct")

    def step(self):
        if self.algorithm <= Algorithm.FP:
            for ts in range(self.config.total_time_slot):
                for cu in self.CUs:
                    # take action
                    cu.set_decision_index(self.dm.take_action())
                    # get reward base on action
                    ts_reward = self.env.cal_reward()
                    # record reward
                    self.reward_record.append(ts_reward)
                    # update environment
                    self.env.step()
        else:
            # MADQL
            print("Under Construct")


if __name__ == "__main__":
    set_logger()
    mn = MobileNetwork()
    # mn.print_CU_position()
    # mn.plot_CU_position()
    # mn.env.print_channel()
    # mn.plot_mobile_network()
    reward = mn.env.cal_reward()
    print(reward)
