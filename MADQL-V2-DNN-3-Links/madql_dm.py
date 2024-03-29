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
    def __init__(self, inputLayer, outputLayer):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(inputLayer, HIDDEN_LAYER[0], bias=True)
        self.hidden_layer1 = nn.Linear(HIDDEN_LAYER[0], HIDDEN_LAYER[1], bias=True)
        self.hidden_layer2 = nn.Linear(HIDDEN_LAYER[1], HIDDEN_LAYER[2], bias=True)
        self.hidden_layer3 = nn.Linear(HIDDEN_LAYER[2], HIDDEN_LAYER[3], bias=True)
        self.output_layer = nn.Linear(HIDDEN_LAYER[3], outputLayer, bias=True)

    def forward(self, x):
        out = F.relu(self.input_layer(x))
        out = F.relu(self.hidden_layer1(out))
        out = F.relu(self.hidden_layer2(out))
        out = F.relu(self.hidden_layer3(out))
        return self.output_layer(out)


class MADQL:
    def __init__(self, loadModel=False):
        self.logger = logging.getLogger()
        self.algorithm = Algorithm.MADQL
        self.epsilon = EPSILON
        self.linkNumber = 3 * CELL_NUMBER
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.DQN = DQN(INPUT_LAYER, OUTPUT_LAYER).to(self.device)
        if loadModel:
            self.logger.info(f"----------------Load Model From {MODEL_PATH}------------------")
            self.DQN.state_dict(torch.load(MODEL_PATH))
        else:
            self.logger.info("----------------Create New Neural Network------------------")
        # set optimizer and loss
        self.optimizer = torch.optim.Adam(self.DQN.parameters(), lr=LEARNING_RATE)
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
            self.logger.info(f'time slot = {self.trainSlot + 1}, average loss = {self.accumulateLoss / PRINT_SLOT}, '
                             f'current epsilon = {self.epsilon}')
            self.accumulateLoss = 0.

    def takeAction(self, channels, trainNetwork=True):
        # build state and forward
        states = np.zeros([self.linkNumber, INPUT_LAYER], dtype=float)
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
        self.train(trainNetwork)

        return actions

    def buildState(self, index, channels):
        state = np.zeros(INPUT_LAYER, dtype=float)
        if index % 3 == 0:
            otherIndex1 = index + 2
            otherIndex2 = index + 1
        elif index % 3 == 1:
            otherIndex1 = index - 1
            otherIndex2 = index + 1
        else:
            otherIndex1 = index - 1
            otherIndex2 = index - 2
        indexes = [index, otherIndex1, otherIndex2]
        count = 0
        for i in range(self.linkNumber):
            for j in range(self.linkNumber):
                for k in range(CODEBOOK_SIZE):
                    channel = channels[generateChannelIndex(indexes[i], indexes[j])].getCSI()
                    beamformer = BEAMFORMER_LIST[k]
                    state[count] = np.linalg.norm(np.matmul(channel, beamformer))
                    count += 1
        return state / np.max(state)

    def train(self, trainNetwork):
        if trainNetwork == False or self.memoryPool.getSize() <= BATCH_SIZE:
            return
        else:
            batch = self.memoryPool.getBatch()
            states = np.concatenate([item[0] for item in batch])
            rewards = np.concatenate([item[1] for item in batch])
            x = torch.from_numpy(states).float().to(self.device)
            y = torch.from_numpy(rewards).float().to(self.device)
            self.optimizer.zero_grad()
            self.DQN.zero_grad()
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
