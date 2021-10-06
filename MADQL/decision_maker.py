import random
from config import Config
from dqn import DQN


class WMMSE:
    def __init__(self):
        self.config = Config()


class FP:
    def __init__(self):
        self.config = Config()


class Random:
    def __init__(self):
        self.config = Config()

    def take_action(self):
        action = []
        for i in range(3):
            """random beamformer and power"""
            codebook_index = random.randint(0, self.config.codebook_size-1)
            power_index = random.randint(0, self.config.power_level-1)
            action.append([codebook_index, power_index])
        return action


class MaxPower:
    def __init__(self):
        self.config = Config()

    def take_action(self):
        action = []
        for i in range(3):
            codebook_index = random.randint(0, self.config.codebook_size-1)
            power_index = self.config.power_level-1     # choose the maximum power
            action.append([codebook_index, power_index])
        return action


class MADQL:
    def __init__(self):
        print("[MADQL] Under Construct!")
        self.target_dqn = DQN()
