import logging
import random

import torch.nn as nn
import torch.optim
import torch

from config import *
from utils import Algorithm


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # layers


class MADQL:
    def __init__(self, loadModel=False):
        self.logger = logging.getLogger()
        self.algorithm = Algorithm.MADQL
        self.epsilon = EPSILON
        """DQN"""
        self.trainDQN = DQN()
        self.targetDQN = DQN()
        if loadModel:
            self.logger.info(f"----------------Load Model From {MODEL_PATH}------------------")
            self.trainDQN.state_dict(torch.load(MODEL_PATH))
        else:
            self.logger.info("----------------Create New Network------------------")
        # copy parameter of train DQN to target DQN
        self.targetDQN.load_state_dict(self.trainDQN.state_dict())
        # set optimizer and loss
        self.optimizer = torch.optim.Adam(self.trainDQN.parameters(), lr=LEARNING_RATE, weight_decay=REG_BETA)
        self.loss = nn.MSELoss()
        # set tensor on GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainDQN.to(self.device)
        self.targetDQN.to(self.device)

    def takeAction(self, index, lastAction, channels, trainNetwork=True):
        if np.random.rand() < self.epsilon and trainNetwork:
            return self.takeActionRandom()
        else:
            # build state
            # forward
            # calculate reward
            # save in memory pool
            # train
            return []

        return []

    def takeActionRandom(self):
        return [random.randint(0, POWER_LEVEL - 1), random.randint(0, CODEBOOK_SIZE - 1)]

    def buildState(self, index, channels):
        return []

    def calReward(self):
        return 0.

    def updateModelParameter(self):
        self.logger.info("----------------Target DQN Parameter Updates!------------------")
        self.logger.info(f"----------------Current Epsilon: {self.epsilon}------------------")
        self.targetDQN.load_state_dict(self.trainDQN.state_dict())

    def saveModel(self):
        self.logger.info(f"----------------Save Model To {MODEL_PATH}------------------")
        torch.save(self.trainDQN.state_dict(), MODEL_PATH)
