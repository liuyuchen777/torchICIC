import copy
import logging
import random

import torch.optim
import torch.nn as nn
import numpy as np

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
        self.logger = logging.getLogger()
        self.trainDQN = DQN()
        self.targetDQN = DQN()
        self.config = Config()
        self.optimizer = torch.optim.Adam(self.trainDQN.parameters(), lr=self.config.learningRate,
                                          weight_decay=self.config.regBeta)
        self.loss = nn.MSELoss()
        self.bn = nn.BatchNorm2d(num_features=1, eps=0, affine=False)
        self.count = 0

    def takeActionRandom(self):
        # random
        action = []
        for i in range(3):
            codebookIndex = random.randint(0, self.config.codebookSize - 1)
            powerIndex = self.config.powerLevel - 1  # choose the maximum power
            action.append([codebookIndex, powerIndex])
        return action

    def takeAction(self, state, previous, trainLabel=True):
        """state is defined in MobileNetwork.buildState"""
        # epsilon-greedy policy
        if np.random.rand() < self.config.epsilon and trainLabel:
            # random
            action = []
            for i in range(3):
                codebookIndex = random.randint(0, self.config.codebookSize - 1)
                powerIndex = self.config.powerLevel - 1  # choose the maximum power
                action.append([codebookIndex, powerIndex])
            return action
        else:
            # greedy
            action = []
            with torch.no_grad():
                flow1 = self.bn(torch.unsqueeze(torch.unsqueeze(torch.tensor(state[0], dtype=torch.float32), 0), 0))
                flow2 = self.bn(torch.unsqueeze(torch.unsqueeze(torch.tensor(state[1], dtype=torch.float32), 0), 0))
                predict = self.targetDQN.forward(flow1, flow2)
            index = torch.argmax(predict).item()
            for i in range(3):
                power = index % 2
                index /= 2
                beamformer = index % 2
                index /= 2
                if power:
                    if np.random.rand() < self.config.keepAlpha:
                        powerIndex = previous[i][1] + 1
                    else:
                        powerIndex = previous[i][1] - 1
                    if powerIndex < 0:
                        powerIndex = 0
                    if powerIndex >= self.config.powerLevel:
                        powerIndex = self.config.powerLevel - 1
                else:
                    powerIndex = previous[i][1]
                if beamformer:
                    if np.random.rand() < self.config.keepAlpha:
                        beamformerIndex = previous[i][0] + 1
                    else:
                        beamformerIndex = previous[i][0] - 1
                    if beamformerIndex < 0:
                        beamformerIndex = 0
                    if beamformerIndex >= self.config.codebookSize:
                        beamformerIndex = self.config.codebookSize - 1
                else:
                    beamformerIndex = previous[i][0]
                action.append([beamformerIndex, powerIndex])
            return action

    def backProp(self, recordBatch):
        """
        record = (s, a, r, s')
        """
        self.count += 1
        # train network
        flow1 = self.bn(torch.unsqueeze(torch.tensor([item[0][0] for item in recordBatch], dtype=torch.float32), 1))
        flow2 = self.bn(torch.unsqueeze(torch.tensor([item[0][1] for item in recordBatch], dtype=torch.float32), 1))
        predictTrain = self.trainDQN.forward(flow1, flow2)
        with torch.no_grad():
            # target network
            flowNext1 = self.bn(torch.unsqueeze(torch.tensor([item[3][0] for item in recordBatch], dtype=torch.float32), 1))
            flowNext2 = self.bn(torch.unsqueeze(torch.tensor([item[3][1] for item in recordBatch], dtype=torch.float32), 1))
            predictTarget = self.targetDQN.forward(flowNext1, flowNext2)
            # revise reward
            rewardMax = torch.amax(predictTarget, dim=1)
            action = [item[1] for item in recordBatch]
            reward = [item[2] for item in recordBatch]
            rewardRevise = torch.tensor(reward) + self.config.gamma * rewardMax
            for i in range(self.config.batchSize):
                predictTarget[i][action[i]] = rewardRevise[i]
        # calculate loss
        myLoss = self.loss(predictTrain, predictTarget)
        myLoss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # copy train model to target model after T_step
        if self.count > self.config.tStep:
            self.count = 0
            self.targetDQN = copy.deepcopy(self.trainDQN)
        # return loss to time slot
        return myLoss
