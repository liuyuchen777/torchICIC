from config import Config
from utils import Algorithm, set_logger
import matplotlib.pyplot as plt
import numpy as np
from coordination_unit import CU
from memory_pool import MemoryPool
from decision_maker import MaxPower, MADQL, Random, FP, WMMSE
from environment import Environment
import logging
import json


def set_decision_maker(algorithm):
    if algorithm == Algorithm.RANDOM:
        return Random()
    elif algorithm == Algorithm.MAX_POWER:
        return MaxPower()
    elif algorithm == Algorithm.FP:
        return FP()
    elif algorithm == Algorithm.WMMSE:
        return WMMSE()
    elif algorithm == Algorithm.MADQL:
        return MADQL()
    else:
        exit(1)


class MobileNetwork:
    def __init__(self, algorithm=Algorithm.RANDOM):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.CUs = []
        self._generate_CUs_()
        self.mp = MemoryPool()
        self.algorithm = algorithm
        self.dm = set_decision_maker(algorithm)
        """generate mobile network"""
        self.env = Environment(self.CUs)
        self.reward_record = []
        self.reward_average_record = []

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
        print("[build_state] Under Construct")

    def get_reward_record(self):
        return self.reward_record

    def get_reward_average_record(self):
        return self.reward_average_record

    def train(self):
        """train network"""
        if self.algorithm == Algorithm.RANDOM or self.algorithm == Algorithm.MAX_POWER:
            for ts in range(self.config.total_time_slot):
                # step
                ts_reward, ts_reward_average = self.step()
                # record reward
                self.reward_record.append(ts_reward)
                self.reward_average_record.append(ts_reward_average)
                # print
                print(f'time slot: {ts + 1}, system average reward: {ts_reward_average}.')
        else:
            # MADQL
            print("[train] Under Construct")

    def step(self):
        """one step"""
        if self.algorithm == Algorithm.RANDOM or self.algorithm == Algorithm.MAX_POWER:
            for cu in self.CUs:
                # take action
                cu.set_decision_index(self.dm.take_action())
            # get reward base on action
            ts_reward = self.env.cal_reward()
            ts_reward_average = sum(ts_reward) / len(ts_reward)
            # update environment
            self.env.step()
            # return
            return ts_reward, ts_reward_average
        else:
            # MADQL
            print("[step] Under Construct")

    def save_rewards(self, name="default"):
        """save rewards record to json file"""
        with open('./data/data.txt') as jsonFile:
            data = json.load(jsonFile)
        with open('./data/data.txt', 'w') as jsonFile:
            data[name + "-rewards"] = self.reward_record
            data[name + "-average-rewards"] = self.reward_average_record
            json.dump(data, jsonFile)

    def load_rewards(self, name="default"):
        """load rewards record to current mn"""
        with open('./data/data.txt') as jsonFile:
            data = json.load(jsonFile)
            self.reward_record = data[name + "-rewards"]
            self.reward_average_record = data[name + "-average-rewards"]


def cdf(x, plot=True, *args, **kwargs):
    x, y = sorted(x), np.arange(len(x)) / len(x)
    plt.plot(x, y, *args, **kwargs) if plot else (x, y)


def print_reward(mn1, mn2):
    mn1.train()
    average_rewards1 = mn1.get_reward_average_record()
    cdf(average_rewards1, label="Random")

    mn2.train()
    average_rewards2 = mn2.get_reward_average_record()
    cdf(average_rewards2, label="Max Power")

    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    set_logger()
    mn1 = MobileNetwork(Algorithm.RANDOM)
    mn2 = MobileNetwork(Algorithm.MAX_POWER)
    """network structure"""
    # mn.plot_mobile_network()
    """print reward distribution"""
    print_reward(mn1, mn2)
    mn1.save_rewards("default-random")
    mn2.save_rewards("default-max-power")
