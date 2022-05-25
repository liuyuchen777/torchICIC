import logging

import torch.nn as nn
import torch.optim
import torch
import torch.nn.functional as F

from config import *
from memory_pool import MemoryPool
from utils import Algorithm, calCapacity, action2Index, index2Action, buildCUIndexList, dBm2num, sigmoid
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
    def __init__(self, loadModel):
        self.logger = logging.getLogger()
        self.algorithm = Algorithm.MADQL
        self.epsilon = EPSILON
        self.linkNumber = 3 * CELL_NUMBER
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"--------------------Device {self.device}-----------------------")
        if loadModel:
            self.logger.info(f"----------------Load Model From {MODEL_PATH}------------------")
            self.DQN = DQN(INPUT_LAYER, OUTPUT_LAYER)
            self.DQN.load_state_dict(torch.load(MODEL_PATH))
            self.DQN.to(self.device)
        else:
            self.logger.info("----------------Create New Neural Network------------------")
            self.DQN = DQN(INPUT_LAYER, OUTPUT_LAYER).to(self.device)
        # set optimizer and loss
        self.optimizer = torch.optim.Adam(self.DQN.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()
        # memory pool
        self.memoryPool = MemoryPool()
        # update network
        self.trainSlot = 0
        self.accumulateLoss = 0.

    def epsilonGreedyPolicy(self, outputs, trainNetwork):
        actions = []
        if np.random.rand() < self.epsilon and trainNetwork:
            actions = takeActionRandom(self.linkNumber)
        else:
            for index in range(self.linkNumber):
                actionIndex = int(np.argmax(outputs[index, :]))
                actions.append(index2Action(actionIndex))
        return actions

    def calReward(self, actions, env):
        capacities = calCapacity(actions, env)
        if CELL_NUMBER > 1:
            rewards = [0. for _ in range(self.linkNumber)]
            for i in range(len(rewards)):
                indexes = buildCUIndexList(i)
                for index in indexes:
                    rewards[i] += capacities[index]
                rewards[i] /= 3
                rewardPenalty = self.calInterferencePenaltySig(actions, env, i)
                rewards[i] = rewards[i] - INTERFERENCE_PENALTY * rewardPenalty
        else:
            """Number of  CU = 1"""
            averageCapacity = sum(capacities) / len(capacities)
            rewards = [averageCapacity for _ in range(self.linkNumber)]
        return rewards

    def calInterferencePenaltySig(self, actions, env, index):
        rewardPenalty = 0.
        beamformer = BEAMFORMER_LIST[actions[index][1]]
        for j in range(self.linkNumber):
            if index == j:
                continue
            else:
                channel = env.getChannel(index, j).getCSI()
                rewardPenalty += np.linalg.norm(np.matmul(channel, beamformer))
        rewardPenalty = sigmoid(rewardPenalty)
        return rewardPenalty

    def calInterferencePenaltyLog(self, actions, env, index):
        """log2"""
        rewardPenalty = 0.
        power = dBm2num(POWER_LIST[actions[index][0]])
        beamformer = BEAMFORMER_LIST[actions[index][1]]
        for j in range(self.linkNumber):
            if index == j:
                continue
            else:
                channel = env.getChannel(index, j).getCSI()
                rewardPenalty += np.log2(1+power * np.linalg.norm(np.matmul(channel, beamformer))/dBm2num(NOISE_POWER))
        rewardPenalty /= self.linkNumber - 1
        return rewardPenalty

    def decreaseEpsilon(self):
        self.epsilon = max(self.epsilon / (1 + EPSILON_DECREASE), EPSILON_MIN)

    def printInformation(self):
        if self.trainSlot % PRINT_SLOT == 0:
            self.logger.info(f'train slot = {self.trainSlot + 1}, average loss = {self.accumulateLoss / PRINT_SLOT}, '
                             f'current epsilon = {self.epsilon}')
            self.accumulateLoss = 0.

    def takeAction(self, env, trainNetwork):
        # build state and forward
        states = np.zeros([self.linkNumber, INPUT_LAYER], dtype=float)
        for index in range(self.linkNumber):
            states[index, :] = self.buildState(index, env)
        with torch.no_grad():
            outputs = self.DQN(torch.from_numpy(states).float().to(self.device)).cpu().detach().numpy()
        # take action
        actions = self.epsilonGreedyPolicy(outputs, trainNetwork)
        if trainNetwork:
            # calculate reward and update Q value
            rewards = self.calReward(actions, env)
            for index in range(self.linkNumber):
                outputs[index, action2Index(actions[index])] = rewards[index]
            states = np.split(states, 3)
            outputs = np.split(outputs, 3)
            self.memoryPool.push([[state, output] for state, output in zip(states, outputs)])
            # train
            self.train()

        return actions

    def buildState(self, index, env):
        """use CSI to build state of link index"""
        state = np.zeros(INPUT_LAYER, dtype=float)
        # local information
        indexes = buildCUIndexList(index)
        count = 0
        for i in range(3):
            for j in range(3):
                for k in range(CODEBOOK_SIZE):
                    channel = env.getChannel(indexes[i], indexes[j]).getCSI()
                    beamformer = BEAMFORMER_LIST[k]
                    state[count] = np.linalg.norm(np.matmul(channel, beamformer))
                    count += 1
        # exchanged information
        if CELL_NUMBER > 1:
            indexList = env.getTopPathLossList(index)
            for otherIndex in indexList:
                for k in range(CODEBOOK_SIZE):
                    channel = env.getChannel(otherIndex, index).getCSI()
                    beamformer = BEAMFORMER_LIST[k]
                    state[count] = np.linalg.norm(np.matmul(channel, beamformer))
                    count += 1
        return state / np.max(state)

    def train(self):
        if self.memoryPool.getSize() > BATCH_SIZE:
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
