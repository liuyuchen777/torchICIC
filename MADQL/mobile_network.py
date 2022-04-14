import json

import pickle

from cu_generator import generateCU
from environment import Environment
from memory_pool import MemoryPool
from coordination_unit import plotCU
from utils import *


class MobileNetwork:

    def __init__(self, algorithm=Algorithm.RANDOM):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.mp = MemoryPool()
        """set decision maker"""
        self.algorithm = algorithm
        self.dm = setDecisionMaker(algorithm)
        """start generate mobile network"""
        self.CUs = generateCU()
        self.env = Environment(self.CUs)
        """end generate mobile network"""
        self.rewardRecord = []                      # 7 * time slot
        self.averageRewardRecord = []               # 1 * time slot
        self.lossRecord = []

    """
    state builder
    """

    def buildStateAP(self, CUIndex):
        """
        build state for DQN input (Amplitude & Phase)

        ver 1.0
        ver 2.0
        """
        # TODO: implement amplitude & phase state input

    def buildStateRI(self, CUIndex):
        """
        build state for DQN input (Real & Image)

        ver 1.0
        1st flow -> 12 * 24 (288)
        2nd flow -> 54 * 8 (432)
        ver 2.0
        total one hot -> (72 + 432) * 1
        72 -> 4 * 9 * 2
        432 -> 6 * (4 * 9 * 2)
        2 = Real + Image
        """
        state = np.zeros(shape=self.config.inputLayer, dtype=float)
        # Intra-CU
        action = self.CUs[CUIndex].getAction()
        for sectorIndex in range(3):
            for otherSectorIndex in range(3):
                index = [CUIndex, sectorIndex, CUIndex, otherSectorIndex]
                channel = self.env.getChannel(index2str(index))
                CSI = channel.getCSI()
                beamformer = self.config.beamformList[action[sectorIndex][0]]
                power = self.config.powerList[action[sectorIndex][1]]
                tmp = dBm2num(power) * beamformer.dot(CSI)
                state[sectorIndex*24+otherSectorIndex*8: sectorIndex*24+otherSectorIndex*8+4] = np.real(tmp)
                state[sectorIndex*24+otherSectorIndex*8+4: sectorIndex*24+otherSectorIndex*8+8] = np.imag(tmp)
        # Inter-CU
        for otherCUIndex in neighborTable[CUIndex]:
            actionHistory = self.CUs[otherCUIndex].getActionHistory()
            for otherSectorIndex in range(3):
                for sectorIndex in range(3):
                    index = [otherCUIndex, otherSectorIndex, CUIndex, sectorIndex]
                    if judgeSkip(index):
                        continue
                    else:
                        channel = self.env.getChannel(index2str(index))
                        CSI = channel.getCSI()  # H_t
                        beamformer = self.config.beamformList[actionHistory[otherSectorIndex][0]]
                        power = self.config.powerList[actionHistory[otherSectorIndex][1]]
                        tmp = dBm2num(power) * beamformer.dot(CSI)
                        state[72+otherCUIndex*72+otherSectorIndex*24+sectorIndex*8:
                              72+otherCUIndex*72+otherSectorIndex*24+sectorIndex*8+4] = np.real(tmp)
                        state[72+otherCUIndex*72+otherSectorIndex*24+sectorIndex*8+4:
                              72+otherCUIndex*72+otherSectorIndex*24+sectorIndex*8+8] = np.imag(tmp)
        return state

    def step(self, trainLabel=True):
        """
        one step in training
        record: <s, a, r, s'>
        """
        if self.algorithm == Algorithm.RANDOM or self.algorithm == Algorithm.MAX_POWER:
            for cu in self.CUs:
                # take action
                action, actionIndex = self.dm.takeAction()
                cu.setAction(action)
            # get reward base on action
            tsReward = self.env.calReward()
            tsAverageReward = sum(tsReward) / len(tsReward)
            # update environment
            self.env.step()
            # return
            return tsReward, tsAverageReward
        elif self.algorithm == Algorithm.CELL_ES:
            # test every action and set cell ES action
            for cu in self.CUs:
                action, actionIndex = self.dm.takeAction(self.env, cu)
                cu.setAction(action)
            # get reward base on action
            tsReward = self.env.calReward()
            tsAverageReward = sum(tsReward) / len(tsReward)
            # update environment
            self.env.step()
            # return
            return tsReward, tsAverageReward
        elif self.algorithm == Algorithm.MADQL:
            stateRecord = []
            actionRecord = []
            for CUIndex in range(self.config.cellNumber):
                # build state
                state = self.buildStateRI(CUIndex)
                stateRecord.append(state)
                # take action
                if self.mp.getSize() < self.config.batchSize:
                    action, actionIndex = self.dm.takeActionRandom(self.CUs[CUIndex].getActionHistory())
                    self.CUs[CUIndex].setAction(action)
                else:
                    action, actionIndex = self.dm.takeAction(state, self.CUs[CUIndex].getActionHistory(),
                                                             trainLabel)
                    self.CUs[CUIndex].setAction(action)
                actionRecord.append(actionIndex)
            # calculate reward
            tsReward = self.env.calReward()
            tsAverageReward = sum(tsReward) / len(tsReward)
            # update env
            self.env.step()
            # next state
            nextStateRecord = []
            for CUIndex in range(self.config.cellNumber):
                # build state & take action
                state = self.buildStateRI(CUIndex)
                nextStateRecord.append(state)
            # save in memory pool
            for CUIndex in range(self.config.cellNumber):
                record = [stateRecord[CUIndex], actionRecord[CUIndex], tsReward[CUIndex], nextStateRecord[CUIndex]]
                self.mp.push(record)
            # return
            return tsReward, tsAverageReward

    def train(self):
        """train network"""
        for ts in range(self.config.totalTimeSlot):
            # step
            tsReward, tsAverageReward = self.step()
            # record reward
            self.rewardRecord.append(tsReward)
            self.averageRewardRecord.append(tsAverageReward)
            if self.algorithm == Algorithm.RANDOM \
                    or self.algorithm == Algorithm.MAX_POWER \
                    or self.algorithm == Algorithm.CELL_ES:
                if ts % self.config.printSlot == 0:
                    self.logger.info(f'[Train] mode: {self.algorithm},time slot: {ts + 1}, '
                                     f'system average reward: {tsAverageReward}')
            elif self.algorithm == Algorithm.MADQL:
                if self.mp.getSize() > self.config.batchSize:
                    loss = self.dm.backProp(self.mp.getBatch())
                    self.lossRecord.append(loss)
                    if ts % self.config.printSlot == 0:
                        # copy DQN parameters
                        self.logger.info(f'[train] mode: {self.algorithm}, time slot: {ts}, '
                                         f'system average reward: {tsAverageReward}, '
                                         f'loss: {loss}.')
                    if ts % self.config.tStep == 0:
                        self.dm.updateModelParameter()
                else:
                    continue
        # finish train
        self.logger.info("training finished")
        self.dm.saveModel()
        self.logger.info("model saved")

    """
    getter & setter
    """

    def getRewardRecord(self):
        return self.rewardRecord

    def getAverageRewardRecord(self):
        return self.averageRewardRecord

    def setRewardRecord(self, reward):
        self.rewardRecord = reward

    def setAverageRewardRecord(self, averageReward):
        self.averageRewardRecord = averageReward

    def setConfig(self, config):
        self.config = config

    def cleanReward(self):
        self.cleanRewardRecord()
        self.cleanAverageRewardRecord()

    def cleanRewardRecord(self):
        self.rewardRecord = []

    def cleanAverageRewardRecord(self):
        self.averageRewardRecord = []

    """
    save data as json
    """

    def saveRewards(self, name="default"):
        """save rewards record to json file"""
        with open('data/reward-data.txt') as jsonFile:
            data = json.load(jsonFile)
        with open('data/reward-data.txt', 'w') as jsonFile:
            data[name + "-rewards"] = self.getRewardRecord()
            data[name + "-average-rewards"] = self.getAverageRewardRecord()
            json.dump(data, jsonFile)

    def saveLoss(self, name="default"):
        with open('data/reward-data.txt') as jsonFile:
            data = json.load(jsonFile)
        with open('data/reward-data.txt', 'w') as jsonFile:
            data[name + "-loss"] = self.lossRecord
            json.dump(data, jsonFile)

    def loadRewards(self, name="default"):
        with open('data/reward-data.txt') as jsonFile:
            data = json.load(jsonFile)
            self.rewardRecord = data[name + "-rewards"]
            self.averageRewardRecord = data[name + "-average-rewards"]


"""
Save and load model structure from json file
"""


def saveMobileNetwork(mn):
    with open('data/mobile-network-data.txt', 'wb') as file:
        pickle.dump(mn, file)


def loadMobileNetwork():
    with open('data/mobile-network-data.txt', 'rb') as file:
        mn = pickle.load(file)
        return mn


"""
network structure visualization
"""


def printCUPosition(mn):
    for i in range(len(mn.CUs)):
        logging.info(f'Position of {i}th CU is: {mn.CUs[i].pos}')


def plotCUPosition(mn):
    CUPosX = []
    CUPosY = []
    for CU in mn.CUs:
        CUPosX.append(CU.pos[0])
        CUPosY.append(CU.pos[1])
    plt.scatter(CUPosX, CUPosY, c='r')
    # show figure
    plt.show()


def plotMobileNetwork(mn):
    for cu in mn.CUs:
        plotCU(cu, plt=plt)
    plt.xlabel(r'$d_x$/m')
    plt.ylabel(r'$d_y$/m')
    plt.title("System Model of MIMO Cellular Network")
    plt.show()
