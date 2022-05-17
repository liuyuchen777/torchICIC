import logging

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
        super().__init__()
        # layers
        self.inputLayer = nn.Conv2d(in_channels=INPUT_CHANNEL, out_channels=16, kernel_size=3, padding=1)
        self.convLayer1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.convLayer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.fullCon = nn.Linear(12 * 12 * 64, 64)
        self.outputLayer = nn.Linear(64, CODEBOOK_SIZE * POWER_LEVEL)

    def forward(self, x):
        x = F.relu(self.inputLayer(x))
        x = F.relu(self.convLayer1(x))
        x = F.relu(self.convLayer2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fullCon(x))
        return F.relu(self.outputLayer(x))


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
            self.targetDQN.state_dict(torch.load(MODEL_PATH))
        else:
            self.logger.info("----------------Create New Neural Network------------------")
        # copy parameter of train DQN to target DQN
        self.targetDQN.load_state_dict(self.trainDQN.state_dict())
        # set optimizer and loss
        self.optimizer = torch.optim.Adam(self.trainDQN.parameters(), lr=LEARNING_RATE)
        self.loss = nn.SmoothL1Loss()
        # set tensor on GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainDQN.to(self.device)
        self.targetDQN.to(self.device)
        # memory pool
        self.memoryPool = MemoryPool()
        # update network
        self.trainSlot = 0
        self.accumulateLoss = 0.

    def takeAction(self, channels, linkNumber, trainNetwork=True):
        states = np.zeros([linkNumber, 2, 12, 12], dtype=float)
        for index in range(linkNumber):
            states[index, :] = self.buildState(index, channels)
        with torch.no_grad():
            outputs = self.targetDQN.forward(torch.from_numpy(states).float().to(self.device)).cpu().detach().numpy()
        actions = []
        if np.random.rand() < self.epsilon and trainNetwork:
            actions = takeActionRandom(linkNumber)
        else:
            for index in range(linkNumber):
                actionIndex = int(np.argmax(outputs[index, :]))
                actions.append(index2Action(actionIndex))
        # calculate capacity
        capacities = calCapacity(actions, channels)
        averageCapacity = sum(capacities) / len(capacities)
        for index in range(linkNumber):
            outputs[index, action2Index(actions[index])] = averageCapacity
        # save in memory pool
        self.memoryPool.push([states, outputs])
        # train
        if trainNetwork and self.memoryPool.getSize() > BATCH_SIZE // (CELL_NUMBER * 3):
            loss = self.backprop()
            self.trainSlot += 1
            self.accumulateLoss += loss.item()
            if self.trainSlot % PRINT_SLOT == 0:
                self.logger.info(f'time slot = {self.trainSlot + 1}, average loss = {self.accumulateLoss / PRINT_SLOT}')
                self.accumulateLoss = 0.
            # Decrease Epsilon
            self.epsilon = max(self.epsilon / (1 + EPSILON_DECREASE), EPSILON_MIN)
            if self.trainSlot % T_STEP == 0:
                self.updateModelParameter()

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
        return state

    def backprop(self):
        batch = self.memoryPool.getBatch()
        states = np.concatenate([item[0] for item in batch])
        outputs = np.concatenate([item[1] for item in batch])
        x = torch.tensor(states, dtype=torch.float32).to(self.device)
        y = torch.tensor(outputs, dtype=torch.float32).to(self.device)
        y_predict = self.trainDQN.forward(x)
        loss = self.loss(y, y_predict)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.trainDQN.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def updateModelParameter(self):
        self.logger.info("----------------Target DQN Parameter Updates!------------------")
        self.logger.info(f"----------------Current Epsilon: {self.epsilon}------------------")
        self.targetDQN.load_state_dict(self.trainDQN.state_dict())

    def saveModel(self):
        self.logger.info(f"----------------Save Model To {MODEL_PATH}------------------")
        torch.save(self.trainDQN.state_dict(), MODEL_PATH)
