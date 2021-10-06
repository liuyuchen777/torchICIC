import random
from Config import Config
from DQN import DQN


class WMMSE:
    def __init__(self):
        self.config = Config()


class FP:
    def __init__(self):
        self.config = Config()


class Random:
    def __init__(self):
        self.config = Config()

    def takeAction(self):
        action = []
        for i in range(3):
            """random beamformer and power"""
            codebookIndex = random.randint(0, self.config.codebookSize - 1)
            powerIndex = random.randint(0, self.config.powerLevel - 1)
            action.append([codebookIndex, powerIndex])
        return action


class MaxPower:
    def __init__(self):
        self.config = Config()

    def takeAction(self):
        action = []
        for i in range(3):
            codebookIndex = random.randint(0, self.config.codebookSize - 1)
            powerIndex = self.config.powerLevel - 1     # choose the maximum power
            action.append([codebookIndex, powerIndex])
        return action


class MADQL:
    def __init__(self):
        print("[MADQL] Under Construct!")
        self.targetDQN = DQN()

    # train
    def train(self):
        print("[train] Under Construct")
        # back prop

    # eval
    def eval(self):
        print("[eval] Under Construct")
        # forward prop
