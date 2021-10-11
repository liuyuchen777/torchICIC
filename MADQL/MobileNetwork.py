from Config import Config
from utils import Algorithm, setLogger
import matplotlib.pyplot as plt
import numpy as np
from CoordinationUnit import CU
from MemoryPool import MemoryPool
from DecisionMaker import MaxPower, MADQL, Random, FP, WMMSE
from Environment import Environment
import logging
import json


def setDecisionMaker(algorithm):
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
        exit(1)


class MobileNetwork:
    def __init__(self, algorithm=Algorithm.RANDOM):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.CUs = []
        self._generateCUs_()
        self.mp = MemoryPool()
        self.algorithm = algorithm
        self.dm = setDecisionMaker(algorithm)
        """generate mobile network"""
        self.env = Environment(self.CUs)
        self.rewardRecord = []
        self.averageRewardRecord = []

    def _generateCUs_(self):
        # max support 7 cells
        CUNumber = self.config.cellNumber
        R = self.config.cellLength * np.sqrt(3)
        # the first CU
        self.CUs.append(CU(0, [0., 0.]))
        for i in range(1, CUNumber):
            theta = np.pi/6 + (i-1)*np.pi/3
            posX = R * np.cos(theta)
            posY = R * np.sin(theta)
            self.CUs.append(CU(i, [posX, posY]))

    def printCUPosition(self):
        for i in range(len(self.CUs)):
            logging.info(f'Position of {i}th CU is: {self.CUs[i].pos}')

    def plot_CU_position(self):
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
        plt.show()

    def buildState(self, CUIndex):
        """
        [CORE]
        build state for DQN input
        two ways: intra-cell + inter-cell
        intra-cell -> 4 * 4 (MIMO CSI) * 3 * 3 -> 12 * 12 = 144 CSI map
        inter-cell -> center CU (12 neighbor sector), edge CU (6 neighbor sector) -> 4 * 1 * 12/6
        p_t-1 * f_t-1 * H_t -> 4 * 1
        """
        # NOTE:
        print("[build_state] Under Construct")
        # 1 CU, 1 record
        state = []
        # 1. build state
        # 1.1. Intra-CU
        intraState = np.zeros(shape=[self.config.BSAntenna*3, self.config.UTAntenna*3], dtype=complex)
        for sectorIndex in range(3):
            for UEIndex in range(3):
                index = [CUIndex, sectorIndex, CUIndex, UEIndex]

        # 1.2. intra

        # 2. return
        return state

    def getRewardRecord(self):
        return self.rewardRecord

    def getAverageRewardRecord(self):
        return self.averageRewardRecord

    def train(self):
        """train network"""
        for ts in range(self.config.totalTimeSlot):
            # step
            tsReward, tsAverageReward = self.step()
            # record reward
            self.rewardRecord.append(tsReward)
            self.averageRewardRecord.append(tsAverageReward)
            # print
            self.logger.info(f'[Train] mode: {self.algorithm},time slot: {ts + 1}, system average reward: {tsAverageReward}.')

    def test(self):
        """evaluate network"""
        for ts in range(self.config.totalTimeSlot):
            # step
            tsReward, tsAverageReward = self.eval()
            # record reward
            self.rewardRecord.append(tsReward)
            self.averageRewardRecord.append(tsAverageReward)
            # print
            self.logger.info(f'[Test] mode: {self.algorithm},time slot: {ts + 1}, system average reward: {tsAverageReward}.')

    def eval(self):
        if self.algorithm == Algorithm.RANDOM or self.algorithm == Algorithm.MAX_POWER:
            self.step()
        elif self.algorithm == Algorithm.MADQL:
            # MADQL eval
            print("[eval] Under Construct")
            # build state
            # dm.eval

    def step(self):
        """one step in training"""
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
            # MADQL
            print("[step] Under Construct")
            for CUIndex in range(self.config.cellNumber):
                # build state
                state = self.buildState(CUIndex)
                self.CUs[CUIndex].setDecisionIndex(self.dm.takeAction(state))
                # build record
                record = []
                # save in memory pool
                self.mp.push(record)
            # if > x times, backprop DQN

            # eval
            self.eval()

    def saveRewards(self, name="default"):
        """save rewards record to json file"""
        with open('./data/data.txt') as jsonFile:
            data = json.load(jsonFile)
        with open('./data/data.txt', 'w') as jsonFile:
            data[name + "-rewards"] = self.rewardRecord
            data[name + "-average-rewards"] = self.averageRewardRecord
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


def showReward(mn1, mn2):
    mn1.train()
    averageRewards1 = mn1.getAverageRewardRecord()
    cdf(averageRewards1, label="Random")

    mn2.train()
    averageRewards2 = mn2.getAverageRewardRecord()
    cdf(averageRewards2, label="Max Power")

    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    setLogger()
    mn1 = MobileNetwork(Algorithm.RANDOM)
    mn2 = MobileNetwork(Algorithm.MAX_POWER)
    """network structure"""
    # mn.plot_mobile_network()
    """print reward distribution"""
    showReward(mn1, mn2)
    mn1.saveRewards("default-random")
    mn2.saveRewards("default-max-power")
