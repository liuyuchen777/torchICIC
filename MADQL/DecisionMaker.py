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
        actionIndex = random.randint(0, self.config.outputLayer - 1)
        for i in range(3):
            """random beamformer and power"""
            codebookIndex = random.randint(0, self.config.codebookSize - 1)
            powerIndex = random.randint(0, self.config.powerLevel - 1)
            action.append([codebookIndex, powerIndex])
        return action, actionIndex


class MaxPower:
    def __init__(self):
        self.config = Config()

    def takeAction(self):
        action = []
        actionIndex = random.randint(0, self.config.outputLayer - 1)
        for i in range(3):
            codebookIndex = random.randint(0, self.config.codebookSize - 1)
            powerIndex = self.config.powerLevel - 1     # choose the maximum power
            action.append([codebookIndex, powerIndex])
        return action, actionIndex


class CellES:
    def __init__(self):
        self.logger = logging.getLogger()
        self.config = Config()

    def takeAction(self, env, cu):
        actionIndex = random.randint(0, self.config.outputLayer - 1)
        bestReward = 0.
        action = []
        # main loop for ES
        for powerIndex0 in range(self.config.powerLevel):
            for beamIndex0 in range(self.config.codebookSize):
                for powerIndex1 in range(self.config.powerLevel):
                    for beamIndex1 in range(self.config.codebookSize):
                        for powerIndex2 in range(self.config.powerLevel):
                            for beamIndex2 in range(self.config.codebookSize):
                                actionTmp = [
                                    [beamIndex0, powerIndex0],
                                    [beamIndex1, powerIndex1],
                                    [beamIndex2, powerIndex2]
                                ]
                                rewardTmp = env.calLocalReward(cu.index, actionTmp)
                                if rewardTmp > bestReward:
                                    bestReward = rewardTmp
                                    action = actionTmp

        return action, actionIndex


class MADQL:
    def __init__(self):
        self.logger = logging.getLogger()
        self.trainDQN = DQN()
        self.targetDQN = DQN()
        self.targetDQN.load_state_dict(self.trainDQN.state_dict())
        self.config = Config()
        self.optimizer = torch.optim.Adam(self.trainDQN.parameters(), lr=self.config.learningRate,
                                          weight_decay=self.config.regBeta)
        self.loss = nn.MSELoss()
        print("PyTorch Version: \n", torch.__version__)

    def takeActionBaseIndex(self, index, previous):
        action = []
        for i in range(3):
            power = index % 3
            index //= 3
            beamformer = index % 3
            index //= 3
            beamformerIndex = previous[i][0]
            powerIndex = previous[i][1]
            # beamformer
            if beamformer == 0 and beamformerIndex != 0:
                beamformerIndex -= 1
            elif beamformer == 2 and beamformerIndex != self.config.codebookSize - 1:
                beamformerIndex += 1
            # power
            if power == 0 and powerIndex != 0:
                powerIndex -= 1
            elif power == 2 and powerIndex != self.config.powerLevel - 1:
                powerIndex += 1
            action.append([beamformerIndex, powerIndex])
        return action

    def takeActionRandom(self, previous):
        # random
        actionIndex = random.randint(0, self.config.outputLayer - 1)
        action = self.takeActionBaseIndex(actionIndex, previous)
        return action, actionIndex

    def takeAction(self, state, previous, trainLabel=True):
        """state is defined in MobileNetwork.buildState"""
        # epsilon-greedy policy
        if np.random.rand() < self.config.epsilon and trainLabel:
            return self.takeActionRandom(previous)
        else:
            with torch.no_grad():
                input = torch.unsqueeze(torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0), 0)
                predict = self.targetDQN.forward(input)
            actionIndex = torch.argmax(predict).item()
            action = self.takeActionBaseIndex(actionIndex, previous)
            return action, actionIndex

    def backProp(self, recordBatch):
        """
        record = (s, a, r, s')
        """
        # train network
        input = torch.tensor([item[0] for item in recordBatch], dtype=torch.float32)
        predictTrain = self.trainDQN.forward(input)
        with torch.no_grad():
            # target network
            inputNext = torch.tensor([item[3] for item in recordBatch], dtype=torch.float32)
            predictTarget = self.targetDQN.forward(inputNext)
            # revise reward
            rewardMax = torch.amax(predictTarget, dim=1)
            action = [item[1] for item in recordBatch]
            reward = [item[2] for item in recordBatch]
            rewardRevise = torch.tensor(reward) + self.config.gamma * rewardMax
            for i in range(self.config.batchSize):
                predictTarget[i][action[i]] = rewardRevise[i]
        # calculate loss
        myLoss = self.loss(predictTarget, predictTrain)
        myLoss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # return loss to time slot
        return myLoss

    def updateModelParamter(self):
        self.logger.info("DQN Parameter Updates!")
        self.targetDQN.load_state_dict(self.trainDQN.state_dict())
