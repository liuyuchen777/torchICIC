import logging

import numpy as np
import torch.nn as nn
import torch.optim
import torch
import torch.nn.functional as F

from config import *
from memory_pool import MemoryPool
from utils import Algorithm, calCapacity, action2Index, index2Action, generateChannelIndex
from random_dm import takeActionRandom


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # layers
        self.inputLayer = nn.Conv2d(in_channels=INPUT_CHANNEL, out_channels=8, kernel_size=3, padding=1)
        self.convLayer1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.convLayer2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.fullCon = nn.Linear(12 * 12 * 8, 64)
        self.outputLayer = nn.Linear(64, CODEBOOK_SIZE * POWER_LEVEL)

    def forward(self, x):
        x = F.sigmoid(self.inputLayer(x))
        x = F.sigmoid(self.convLayer1(x))
        x = F.sigmoid(self.convLayer2(x))
        x = torch.flatten(x, 1)
        x = F.sigmoid(self.fullCon(x))
        return F.relu(self.outputLayer(x))


class MADQL:
    def __init__(self, loadModel=False):
        self.logger = logging.getLogger()
        self.algorithm = Algorithm.MADQL
        self.epsilon = EPSILON
        self.linkNumber = 3 * CELL_NUMBER
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.DQN = DQN().to(self.device)
        if loadModel:
            self.logger.info(f"----------------Load Model From {MODEL_PATH}------------------")
            self.DQN.state_dict(torch.load(MODEL_PATH))
        else:
            self.logger.info("----------------Create New Neural Network------------------")
        # set optimizer and loss
        self.optimizer = torch.optim.RMSprop(self.DQN.parameters())
        self.loss = nn.MSELoss()
        # memory pool
        self.memoryPool = MemoryPool()
        # update network
        self.trainSlot = 0
        self.accumulateLoss = 0.

    def epsilonGreedyPolicy(self, outputs, trainNetwork=True):
        actions = []
        if np.random.rand() < self.epsilon and trainNetwork:
            actions = takeActionRandom(self.linkNumber)
        else:
            for index in range(self.linkNumber):
                actionIndex = int(np.argmax(outputs[index, :]))
                actions.append(index2Action(actionIndex))
        return actions

    def calReward(self, actions, channels):
        capacities = calCapacity(actions, channels)
        return sum(capacities) / len(capacities)

    def decreaseEpsilon(self):
        self.epsilon = max(self.epsilon / (1 + EPSILON_DECREASE), EPSILON_MIN)

    def printInformation(self):
        if self.trainSlot % PRINT_SLOT == 0:
            self.logger.info(f'time slot = {self.trainSlot + 1}, average loss = {self.accumulateLoss / PRINT_SLOT}')
            self.accumulateLoss = 0.

    def takeAction(self, channels, trainNetwork=True):
        # build state and forward
        states = np.zeros([self.linkNumber, 2, 12, 12], dtype=float)
        for index in range(self.linkNumber):
            states[index, :] = self.buildState(index, channels)
        with torch.no_grad():
            outputs = self.DQN(torch.from_numpy(states).float().to(self.device)).cpu().detach().numpy()
        # take action
        actions = self.epsilonGreedyPolicy(outputs, trainNetwork)
        # calculate reward and update Q value
        reward = self.calReward(actions, channels)
        for index in range(self.linkNumber):
            outputs[index, action2Index(actions[index])] = reward
        states = np.split(states, 3)
        outputs = np.split(outputs, 3)
        self.memoryPool.push([[state, output] for state, output in zip(states, outputs)])
        # train
        self.train()

        return actions

    def buildState(self, index, channels):
        state = np.zeros([INPUT_CHANNEL, 12, 12], dtype=float)
        if index % 3 == 0:
            upIndex = index + 2
            downIndex = index + 1
        elif index % 3 == 1:
            upIndex = index - 1
            downIndex = index + 1
        else:
            upIndex = index - 1
            downIndex = index - 2
        indexes = [upIndex, index, downIndex]
        for i in range(3):
            for j in range(3):
                channel = channels[generateChannelIndex(indexes[i], indexes[j])].getCSI()
                state[0, i*4:i*4+4, j*4:j*4+4] = np.real(channel)
                state[1, i*4:i*4+4, j*4:j*4+4] = np.imag(channel)
        state[0, :] = state[0, :] / np.max(state[0, :])
        state[1, :] = state[1, :] / np.max(state[1, :])
        return state

    def train(self):
        if self.memoryPool.getSize() < BATCH_SIZE:
            return
        else:
            batch = self.memoryPool.getBatch()
            states = np.concatenate([item[0] for item in batch])
            rewards = np.concatenate([item[1] for item in batch])
            x = torch.from_numpy(states).float().to(self.device)
            y = torch.from_numpy(rewards).float().to(self.device)
            self.optimizer.zero_grad()
            y_predict = self.DQN(x)
            loss = self.loss(y_predict, y)
            loss.backward()
            self.optimizer.step()
            # log and add
            self.trainSlot += 1
            self.accumulateLoss += loss.item()
            self.printInformation()
            self.decreaseEpsilon()

    def saveModel(self):
        self.logger.info(f"----------------Save Model To {MODEL_PATH}------------------")
        torch.save(self.DQN.state_dict(), MODEL_PATH)
