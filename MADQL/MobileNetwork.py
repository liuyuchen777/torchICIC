import json
import logging

import matplotlib.pyplot as plt
import numpy as np

from CoordinationUnit import CU
from DecisionMaker import MaxPower, MADQL, Random, FP, WMMSE
from Environment import Environment
from MemoryPool import MemoryPool
from utils import *


class MobileNetwork:
    def __init__(self, algorithm=Algorithm.RANDOM):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.CUs = []
        self._generateCUs_()
        self.mp = MemoryPool()
        self.algorithm = algorithm
        self.dm = self.setDecisionMaker(algorithm)
        """generate mobile network"""
        self.env = Environment(self.CUs)
        self.rewardRecord = []
        self.averageRewardRecord = []
        self.lossRecord = []
        self.count = 0
        self.tmpLossSum = 0.
        self.tmpRewardSum = 0.
        self.epoch = 0

    def _generateCUs_(self):
        # max support 7 cells
        CUNumber = self.config.cellNumber
        R = self.config.cellSize * np.sqrt(3)
        # the first CU
        self.CUs.append(CU(0, [0., 0.]))
        for i in range(1, CUNumber):
            theta = np.pi / 6 + (i - 1) * np.pi / 3
            posX = R * np.cos(theta)
            posY = R * np.sin(theta)
            self.CUs.append(CU(i, [posX, posY]))

    def setDecisionMaker(self, algorithm):
        if algorithm == Algorithm.RANDOM:
            return Random()
        elif algorithm == Algorithm.MAX_POWER:
            return MaxPower()
        elif algorithm == Algorithm.FP:
            return FP()
        elif algorithm == Algorithm.WMMSE:
            return WMMSE()
        elif algorithm == Algorithm.MADQL:
            return MADQL()
        else:
            self.logger.error("!!!Wrong Algorithm ID!!!")
            exit(1)

    def printCUPosition(self):
        for i in range(len(self.CUs)):
            logging.info(f'Position of {i}th CU is: {self.CUs[i].pos}')

    def plotCUPosition(self):
        CUPosX = []
        CUPosY = []
        for CU in self.CUs:
            CUPosX.append(CU.pos[0])
            CUPosY.append(CU.pos[1])
        plt.scatter(CUPosX, CUPosY, c='r')
        # show figure
        plt.show()

    def plotMobileNetwork(self):
        for cu in self.CUs:
            cu.plotCU(plt=plt)
        plt.xlabel(r'$d_x$/m')
        plt.ylabel(r'$d_y$/m')
        plt.title("System Model of MIMO Cellular Network")
        plt.show()

    def buildStateAP(self, CUIndex):
        """
        build state for DQN input (Amplitude & Phase)
        two parts: intra-cell + inter-cell
        intra-cell -> 4 * 4 (MIMO CSI) * 2 (complex -> double) * 3 * 3 -> 12 * 24 = 288 CSI map
        single inter-cell item -> p_t-1 * f_t-1 * H_t -> 4 * 1 * 2
        inter-cell -> center CU (12 neighbor sector), edge CU (6 neighbor sector)
                   -> 4 * 1 * 2 (complex -> double) * 18 * 3 * 3
                   -> 63 * 8
        """

    def buildStateRI(self, CUIndex):
        """
        build state for DQN input (Real & Image)

        """
        # 1 CU, 1 record
        state = []
        # 1. build st  ate
        # 1.1. Intra-CU
        intraState = np.zeros(shape=[self.config.BSAntenna * 3, self.config.UTAntenna * 3 * 2], dtype=float)
        for sectorIndex in range(3):
            for otherSectorIndex in range(3):
                index = [CUIndex, sectorIndex, CUIndex, otherSectorIndex]
                channel = self.env.getChannel(index2str(index))
                CSI = channel.getCSI()
                real = np.real(CSI)
                image = np.imag(CSI)
                # put real part in
                intraState[sectorIndex * self.config.BSAntenna:(sectorIndex + 1) * self.config.BSAntenna,
                otherSectorIndex * 2 * self.config.UTAntenna:(otherSectorIndex * 2 + 1) * self.config.UTAntenna] = real
                # put image part in
                intraState[sectorIndex * self.config.BSAntenna:(sectorIndex + 1) * self.config.BSAntenna,
                    (otherSectorIndex * 2 + 1) * self.config.UTAntenna:(otherSectorIndex * 2 + 2)
                    * self.config.UTAntenna] = image
        # 1.2. intra
        interState = np.zeros(shape=[3 * 3 * self.config.cellNumber, self.config.UTAntenna * 2], dtype=float)
        for otherCUIndex in neighborTable[CUIndex]:
            decisionIndexHistory = self.CUs[otherCUIndex].getDecisionIndexHistory()
            for otherSectorIndex in range(3):
                for sectorIndex in range(3):
                    index = [CUIndex, sectorIndex, otherCUIndex, otherSectorIndex]
                    if judgeSkip(index):
                        continue
                    else:
                        channel = self.env.getChannel(index2str(index))
                        CSI = channel.getCSI()  # H_t
                        beamformer = self.config.beamformList[decisionIndexHistory[sectorIndex][0]]
                        power = self.config.powerList[decisionIndexHistory[sectorIndex][1]]
                        tmp = dBm2num(power) * beamformer.dot(CSI)
                        item = np.zeros(shape=[1, self.config.UTAntenna * 2], dtype=float)
                        item[0, 0:self.config.UTAntenna] = np.real(tmp)
                        item[0, self.config.UTAntenna:] = np.imag(tmp)
                        interState[sectorIndex * (3 * self.config.cellNumber)
                                   + otherCUIndex * 3 + otherSectorIndex, :] = item
        # 2. build state & return
        state.append(intraState)
        state.append(interState)
        return state

    def getRewardRecord(self):
        return self.rewardRecord

    def getAverageRewardRecord(self):
        return self.averageRewardRecord

    def cleanRewardRecord(self):
        self.rewardRecord = []

    def cleanAverageRewardRecord(self):
        self.averageRewardRecord = []

    def train(self):
        """train network"""
        for ts in range(self.config.totalTimeSlot):
            # step
            tsReward, tsAverageReward = self.step()
            # record reward
            self.rewardRecord.append(tsReward)
            self.averageRewardRecord.append(tsAverageReward)
            if self.algorithm == Algorithm.RANDOM or self.algorithm == Algorithm.MAX_POWER:
                self.logger.info(
                    f'[Train] mode: {self.algorithm},time slot: {ts + 1}, system average reward: {tsAverageReward}')
            elif self.algorithm == Algorithm.MADQL:
                if self.mp.getSize() > self.config.batchSize:
                    loss = self.dm.backProp(self.mp.getBatch())
                    self.lossRecord.append(loss)
                    self.tmpRewardSum += tsAverageReward
                    self.tmpLossSum += loss
                    self.count += 1
                    if self.count > self.config.tStep:
                        # print train log
                        self.logger.info(f'[train] mode: {self.algorithm}, epoch: {self.epoch + 1}, '
                                         f'system average reward: {self.tmpRewardSum / self.config.tStep}, '
                                         f'loss: {self.tmpLossSum / self.config.tStep}.')
                        # eval
                        evalReward = self.eval(self.config.evalTimes)
                        self.logger.info(f'[eval] evaluation reward: {evalReward}.')
                        self.count = 0
                        self.tmpLossSum = 0.
                        self.tmpRewardSum = 0.
                        self.epoch += 1

    def eval(self, times):
        if self.algorithm == Algorithm.RANDOM or self.algorithm == Algorithm.MAX_POWER:
            return self.step()
        elif self.algorithm == Algorithm.MADQL:
            totalReward = 0.
            for i in range(times):
                _, averageReward = self.step(trainLabel=False)
                totalReward += averageReward
            return totalReward / times

    def step(self, trainLabel=True):
        """
        one step in training
        record: <s, a, r, s'>
        """
        if self.algorithm == Algorithm.RANDOM or self.algorithm == Algorithm.MAX_POWER:
            for cu in self.CUs:
                # take action
                cu.setDecisionIndex(self.dm.takeAction())
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
                    self.CUs[CUIndex].setDecisionIndex(self.dm.takeActionRandom())
                else:
                    self.CUs[CUIndex].setDecisionIndex(self.dm.takeAction(state,
                        self.CUs[CUIndex].getDecisionIndexHistory(), trainLabel))
                action = 0
                decisionIndex = self.CUs[CUIndex].getDecisionIndex()
                decisionIndexHistory = self.CUs[CUIndex].getDecisionIndexHistory()
                # convert action index
                for i in range(3):
                    if decisionIndex[i][0] != decisionIndexHistory[i][0]:
                        action += 1 << (i * 2)
                    if decisionIndex[i][1] != decisionIndexHistory[i][1]:
                        action += 1 << (i * 2 + 1)
                actionRecord.append(action)
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

    def saveRewards(self, name="default"):
        """save rewards record to json file"""
        with open('./data/data.txt') as jsonFile:
            data = json.load(jsonFile)
        with open('./data/data.txt', 'w') as jsonFile:
            data[name + "-rewards"] = self.rewardRecord
            data[name + "-average-rewards"] = self.averageRewardRecord
            json.dump(data, jsonFile)

    def saveLoss(self, name="default"):
        with open('./data/data.txt') as jsonFile:
            data = json.load(jsonFile)
        with open('./data/data.txt', 'w') as jsonFile:
            data[name + "-loss"] = self.lossRecord
            json.dump(data, jsonFile)

    def loadRewards(self, name="default"):
        """load rewards record to current mn"""
        with open('./data/data.txt') as jsonFile:
            data = json.load(jsonFile)
            self.rewardRecord = data[name + "-rewards"]
            self.averageRewardRecord = data[name + "-average-rewards"]


def cdf(x, plot=True, *args, **kwargs):
    x, y = sorted(x), np.arange(len(x)) / len(x)
    plt.plot(x, y, *args, **kwargs) if plot else (x, y)


def showReward(mn):
    mn.setDecisionMaker(Algorithm.RANDOM)
    mn.train()
    averageRewards1 = mn.getAverageRewardRecord()
    cdf(averageRewards1, label="Random")

    mn.cleanRewardRecord()
    mn.cleanAverageRewardRecord()

    mn.algorithm = Algorithm.MAX_POWER
    mn.dm = mn.setDecisionMaker(mn.algorithm)
    mn.train()
    averageRewards2 = mn.getAverageRewardRecord()
    cdf(averageRewards2, label="Max Power")

    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    setLogger()
    """[test] network structure"""
    # mn = MobileNetwork()
    # mn.plotMobileNetwork()
    """[test] reward in random and max power"""
    # mn = MobileNetwork()
    # showReward(mn)
    """[test] build state and build record"""
    mn = MobileNetwork(Algorithm.MADQL)
    mn.train()
