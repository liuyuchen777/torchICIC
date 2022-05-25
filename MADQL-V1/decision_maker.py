import logging
import random

import torch.optim
import torch.nn as nn

from config import *
from dqn import DQN


def takeActionBaseIndex(index, previous):
    action = []

    for sector in range(3):
        power = index % 3
        index //= 3
        beamformer = index % 3
        index //= 3
        beamformerIndex = previous[sector][0]
        powerIndex = previous[sector][1]
        # beamformer
        if beamformer == 0:
            beamformerIndex -= 1
            if beamformerIndex < 0:
                beamformerIndex = CODEBOOK_SIZE - 1
        elif beamformer == 2:
            beamformerIndex += 1
            if beamformerIndex >= CODEBOOK_SIZE:
                beamformerIndex = 0
        # power
        if power == 0:
            powerIndex -= 1
            if powerIndex < 0:
                powerIndex = POWER_LEVEL - 1
        elif power == 2:
            powerIndex += 1
            if powerIndex >= POWER_LEVEL:
                powerIndex = 0
        # append single sector action to action
        action.append([beamformerIndex, powerIndex])

    return action


class WMMSE:
    def __init__(self):
        print("---------Under Construct--------")


class FP:
    def __init__(self):
        print("---------Under Construct--------")


class Random:
    def takeAction(self):
        action = []
        actionIndex = random.randint(0, OUTPUT_LAYER - 1)
        for _ in range(3):
            """random beamformer and power"""
            codebookIndex = random.randint(0, CODEBOOK_SIZE - 1)
            powerIndex = random.randint(0, POWER_LEVEL - 1)
            action.append([codebookIndex, powerIndex])
        return action, actionIndex


class MaxPower:
    def takeAction(self):
        action = []
        actionIndex = random.randint(0, OUTPUT_LAYER - 1)
        for _ in range(3):
            codebookIndex = random.randint(0, CODEBOOK_SIZE - 1)
            powerIndex = POWER_LEVEL - 1     # choose the maximum power
            action.append([codebookIndex, powerIndex])
        return action, actionIndex


class CellES:
    def takeAction(self, env, cu):
        actionIndex = random.randint(0, OUTPUT_LAYER - 1)
        bestReward = 0.
        action = []
        # main loop for ES
        for powerIndex0 in range(POWER_LEVEL):
            for beamIndex0 in range(CODEBOOK_SIZE):
                for powerIndex1 in range(POWER_LEVEL):
                    for beamIndex1 in range(CODEBOOK_SIZE):
                        for powerIndex2 in range(POWER_LEVEL):
                            for beamIndex2 in range(CODEBOOK_SIZE):
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


"""
MADQL Algorithm Agent
"""


class MADQL:
    def __init__(self, mode="NEW"):
        """
        init MADQL decision maker and DQN network
        Args:
            name: NEW or LOAD_MODEL
        """
        self.logger = logging.getLogger()
        # target network and train network, update target to train every T_STEP
        self.trainDQN = DQN()
        self.targetDQN = DQN()
        if mode == "LOAD_MODEL":
            self.logger.info(f"----------------Load Model From {MODEL_PATH}------------------")
            self.trainDQN.state_dict(torch.load(MODEL_PATH))
        elif mode == "NEW":
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
        # greedy-epsilon
        self.epsilon = EPSILON
        # check version
        print("PyTorch Version: \n", torch.__version__)
        print("GPU Device: \n", self.device)

    def takeActionRandom(self, previous):
        # random
        actionIndex = random.randint(0, OUTPUT_LAYER - 1)
        action = takeActionBaseIndex(actionIndex, previous)
        return action, actionIndex

    def takeAction(self, state, previousValue, train=True):
        """state is defined in MobileNetwork.buildState"""
        # epsilon-greedy policy
        if np.random.rand() < self.epsilon and train:
            # random choose
            return self.takeActionRandom(previousValue)
        else:
            # choose optimal
            self.epsilon = self.epsilon * DECREASE_FACTOR
            with torch.no_grad():
                network_input = torch.unsqueeze(torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0), 0).to(self.device)
                predict = self.targetDQN.forward(network_input)
            actionIndex = torch.argmax(predict).item()
            action = takeActionBaseIndex(actionIndex, previousValue)
            return action, actionIndex

    def backProp(self, recordBatch):
        """
        record = (s, a, r, s')
        """
        # train network
        state = torch.tensor([item[0] for item in recordBatch], dtype=torch.float32)
        state = state.to(self.device)
        predictTrain = self.trainDQN.forward(state)
        with torch.no_grad():
            # target network
            stateNext = torch.tensor([item[3] for item in recordBatch], dtype=torch.float32)
            stateNext = stateNext.to(self.device)
            predictTarget = self.targetDQN.forward(stateNext)
            # revise reward
            rewardMax = torch.amax(predictTarget, dim=1)
            action = [item[1] for item in recordBatch]
            reward = [item[2] for item in recordBatch]
            rewardRevise = torch.tensor(reward).to(self.device) + GAMMA * rewardMax
            for i in range(BATCH_SIZE):
                predictTarget[i][action[i]] = rewardRevise[i]
        # calculate loss
        myLoss = self.loss(predictTarget, predictTrain)
        myLoss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        # return loss to time slot
        return myLoss

    def updateModelParameter(self):
        self.logger.info("----------------Target DQN Parameter Updates!------------------")
        self.logger.info(f"----------------Current Epsilon: {self.epsilon}------------------")
        self.targetDQN.load_state_dict(self.trainDQN.state_dict())

    def saveModel(self):
        self.logger.info(f"----------------Save Model To {MODEL_PATH}------------------")
        torch.save(self.trainDQN.state_dict(), MODEL_PATH)
