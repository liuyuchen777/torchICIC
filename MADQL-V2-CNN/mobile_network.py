import logging

from config import *
from utils import Algorithm, calCapacity, saveData
from descision_maker import setDecisionMaker
from mobile_network_generator import generateMobileNetwork, loadMobileNetwork, plotMobileNetwork, saveMobileNetwork
from channel_generator import generateChannel


class MobileNetwork:
    def __init__(self, loadNetwork="default", newNetwork=False, decisionMaker=Algorithm.RANDOM, loadModel=False):
        self.logger = logging.getLogger()
        if loadNetwork != "default" and newNetwork == False:
            """load sector/UE position from local file"""
            self.sectors, self.UEs = loadMobileNetwork(loadNetwork)
        else:
            self.sectors, self.UEs = generateMobileNetwork()
        self.channels = generateChannel(self.sectors, self.UEs)
        self.dm = setDecisionMaker(decisionMaker, loadModel)
        self.accumulateCapacity = 0.
        self.capacity = []                                      # number of links * time slot
        self.averageCapacity = []                               # 1 * time slot
        self.actionHistory = []                                 # 2 * number of links * time slot
        saveMobileNetwork(self.sectors, self.UEs, name=loadNetwork)

    def getSectors(self):
        return self.sectors

    def getUEs(self):
        return self.UEs

    def getCapacity(self):
        return self.capacity

    def getAverageCapacity(self):
        return self.averageCapacity

    def clearRecord(self):
        self.capacity = []
        self.averageCapacity = []
        self.actionHistory = []

    def updateChannel(self):
        for channel in self.channels.values():
            channel.update()

    def saveRecord(self, prefix="default-"):
        self.logger.info(f"--------------------------Save Rewards as {prefix}-----------------------------")
        saveData(self.capacity, name=prefix+"capacity")
        saveData(self.averageCapacity, name=prefix+"averageCapacity")
        saveData(self.actionHistory, name=prefix+"action")

    def step(self):
        for ts in range(TOTAL_TIME_SLOT):
            """take action"""
            actions = []
            if self.dm.algorithm == Algorithm.RANDOM or self.dm.algorithm == Algorithm.MAX_POWER:
                actions = self.dm.takeAction()
            elif self.dm.algorithm == Algorithm.CELL_ES:
                actions = self.dm.takeAction(self.channels)     # CELL_ES only work when CELL_NUMBER is 1
            elif self.dm.algorithm == Algorithm.MADQL:
                actions = self.dm.takeAction(self.channels, len(self.sectors))
            """calculate capacity"""
            currentCapacity = calCapacity(actions, self.channels)
            """record"""
            self.actionHistory.append(actions)
            self.capacity.append(currentCapacity)
            averageCapacity = sum(currentCapacity) / len(currentCapacity)
            self.averageCapacity.append(averageCapacity)
            """update"""
            self.updateChannel()
            """print log"""
            if ts != 0 and ts % PRINT_SLOT == 0:
                self.logger.info(f'mode: {self.dm.algorithm}, time slot: {ts + 1}, '
                                 f'system average capacity: {self.accumulateCapacity / PRINT_SLOT}, '
                                 f'action: {actions}')
                self.accumulateCapacity = 0.
            self.accumulateCapacity += averageCapacity
        """save reward"""
        self.saveRecord(prefix=str(self.dm.algorithm)+"-")
        """save model"""
        if self.dm.algorithm == Algorithm.MADQL:
            self.dm.saveModel()


if __name__ == "__main__":
    mn = MobileNetwork()
    plotMobileNetwork(mn.getSectors(), mn.getUEs())
