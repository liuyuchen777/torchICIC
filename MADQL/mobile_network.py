import json

import pickle

from cu_generator import generateCU
from environment import Environment
from memory_pool import MemoryPool
from coordination_unit import plotCU
from utils import *

from config import *


class MobileNetwork:

    def __init__(self, algorithm=Algorithm.RANDOM):
        self.logger = logging.getLogger(__name__)
        self.mp = MemoryPool()
        """set decision maker"""
        self.algorithm = algorithm
        self.dm = setDecisionMaker(algorithm)
        """start generate mobile network"""
        self.CUs = generateCU()
        self.env = Environment(self.CUs)
        """end generate mobile network"""
        self.rewardRecord = []              # scale: 7 * time slot
        self.averageRewardRecord = []       # scale: 1 * time slot
        self.lossRecord = []
        """tmp variable"""
        self.accumulateLoss = 0.
        self.accumulateAverageReward = 0.

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
        state = np.zeros(shape=INPUT_LAYER, dtype=float)
        # Intra-CU
        action = self.CUs[CUIndex].getAction()
        for sectorIndex in range(3):
            for otherSectorIndex in range(3):
                index = [CUIndex, sectorIndex, CUIndex, otherSectorIndex]
                channel = self.env.getChannel(index2str(index))
                CSI = channel.getCSI()
                beamformer = BEAMFORMER_LIST[action[sectorIndex][0]]
                power = POWER_LIST[action[sectorIndex][1]]
                tmp = dBm2num(power) * beamformer.dot(CSI)
                state[sectorIndex * 24 + otherSectorIndex * 8: sectorIndex * 24 + otherSectorIndex * 8 + 4] = np.real(
                    tmp)
                state[
                sectorIndex * 24 + otherSectorIndex * 8 + 4: sectorIndex * 24 + otherSectorIndex * 8 + 8] = np.imag(tmp)
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
                        beamformer = BEAMFORMER_LIST[actionHistory[otherSectorIndex][0]]
                        power = POWER_LIST[actionHistory[otherSectorIndex][1]]
                        tmp = dBm2num(power) * beamformer.dot(CSI)
                        state[72 + otherCUIndex * 72 + otherSectorIndex * 24 + sectorIndex * 8:
                              72 + otherCUIndex * 72 + otherSectorIndex * 24 + sectorIndex * 8 + 4] = np.real(tmp)
                        state[72 + otherCUIndex * 72 + otherSectorIndex * 24 + sectorIndex * 8 + 4:
                              72 + otherCUIndex * 72 + otherSectorIndex * 24 + sectorIndex * 8 + 8] = np.imag(tmp)
        return state

    def step(self, train=True):
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
            for CUIndex in range(CELL_NUMBER):
                # build state
                state = self.buildStateRI(CUIndex)
                stateRecord.append(state)
                # take action
                if self.mp.getSize() < BATCH_SIZE and train:
                    # memory pool size smaller than batch size
                    action, actionIndex = self.dm.takeActionRandom(self.CUs[CUIndex].getActionHistory())
                    self.CUs[CUIndex].setAction(action)
                else:
                    # greedy-epsilon or greedy -> depends on train or test
                    action, actionIndex = self.dm.takeAction(state, self.CUs[CUIndex].getActionHistory(), train)
                    self.CUs[CUIndex].setAction(action)
                actionRecord.append(actionIndex)
            # calculate reward
            tsReward = self.env.calReward()
            tsAverageReward = sum(tsReward) / len(tsReward)
            # update env
            self.env.step()
            # next state
            nextStateRecord = []
            for CUIndex in range(CELL_NUMBER):
                # build state & take action
                state = self.buildStateRI(CUIndex)
                nextStateRecord.append(state)
            # save in memory pool
            for CUIndex in range(CELL_NUMBER):
                record = [stateRecord[CUIndex], actionRecord[CUIndex], tsReward[CUIndex], nextStateRecord[CUIndex]]
                self.mp.push(record)
            # return
            return tsReward, tsAverageReward

    def train(self, train=True):
        """train network"""
        for ts in range(TOTAL_TIME_SLOT):
            # step
            tsReward, tsAverageReward = self.step(train=train)
            # record reward
            self.rewardRecord.append(tsReward)
            self.averageRewardRecord.append(tsAverageReward)

            if self.algorithm == Algorithm.MADQL and self.mp.getSize() > BATCH_SIZE and train:
                loss = self.dm.backProp(self.mp.getBatch())
                self.lossRecord.append(loss)
                self.accumulateLoss += loss
                self.accumulateAverageReward += tsAverageReward
                if ts % PRINT_SLOT == 0 and ts != 0:
                    self.logger.info(f'[Train] mode: {self.algorithm}, time slot: {ts}, '
                                     f'system average reward: {self.accumulateAverageReward / PRINT_SLOT}, '
                                     f'loss: {self.accumulateLoss / PRINT_SLOT}.')
                    self.accumulateLoss = 0.
                    self.accumulateAverageReward = 0.
                # update target DQN parameters
                if ts % T_STEP == 0 and ts != 0:
                    self.dm.updateModelParameter()
            else:
                self.accumulateAverageReward += tsAverageReward
                if ts % PRINT_SLOT == 0 and ts != 0:
                    self.logger.info(f'[Train] mode: {self.algorithm}, time slot: {ts}, '
                                     f'system average reward: {self.accumulateAverageReward / PRINT_SLOT}')
                    self.accumulateAverageReward = 0.
        # finish train
        self.logger.info("-----------------------------Training Finished------------------------------------")
        if self.algorithm == Algorithm.MADQL:
            self.dm.saveModel()
            self.logger.info("------------------------------Model Saved---------------------------------------")

    """
    cleaner
    """

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
            self.logger.info(f"--------------------------Save Rewards as {name}-----------------------------")
            data[name + "-rewards"] = self.rewardRecord
            data[name + "-average-rewards"] = self.averageRewardRecord
            json.dump(data, jsonFile)

    def saveLoss(self, name="default"):
        with open('data/loss-data.txt') as jsonFile:
            data = json.load(jsonFile)
        with open('data/loss-data.txt', 'w') as jsonFile:
            self.logger.info(f"--------------------------Load Rewards {name}-----------------------------")
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
        pos_and_channel_data = {'pos': mn.CUs, 'channel': mn.env}
        pickle.dump(pos_and_channel_data, file)


def loadMobileNetwork():
    with open('data/mobile-network-data.txt', 'rb') as file:
        pos_and_channel_data = pickle.load(file)
        mn = MobileNetwork()
        mn.CUs = pos_and_channel_data['pos']
        mn.env = pos_and_channel_data['channel']
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
