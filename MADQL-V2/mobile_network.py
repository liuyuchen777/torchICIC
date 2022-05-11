import logging
from datetime import datetime

from config import *
from utils import Algorithm, calCapacity
from descision_maker import setDecisionMaker
from mobile_network_generator import generateMobileNetwork, loadMobileNetwork, plotMobileNetwork
from channel_generator import generateChannel


class MobileNetwork:
    def __init__(self, loadNetwork=False, decisionMaker=Algorithm.RANDOM, loadModel=False):
        self.logger = logging.getLogger()
        if loadNetwork:
            """load sector/UE position from local file"""
            self.sectors, self.UEs = loadMobileNetwork()
        else:
            self.sectors, self.UEs = generateMobileNetwork()
        self.channels = generateChannel(self.sectors, self.UEs)
        self.dm = setDecisionMaker(decisionMaker, loadModel)
        self.capacity = []                                      # number of links * time slot
        self.averageCapacity = []                               # 1 * time slot
        self.actionHistory = []                                 # 2 * number of links * time slot

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

    # def saveRecord(self, prefix="SimulationData-" + datetime.now().strftime("%m/%d/%Y-%H:%M:%S") + "-"):
    #

    def step(self):
        for ts in range(TOTAL_TIME_SLOT):
            """take action"""
            actions = []
            if self.dm.algorithm == Algorithm.RANDOM or self.dm.algorithm == Algorithm.MAX_POWER:
                for _ in range(len(self.sectors)):
                    actions.append(self.dm.takeAction())
            elif self.dm.algorithm == Algorithm.CELL_ES:
                """CELL_ES only work when CELL_NUMBER is 1"""
                actions.extend(self.dm.takeAction(self.channels))
            elif self.dm.algorithm == Algorithm.MADQL:
                for i in range(len(self.sectors)):
                    actions.append(self.dm.takeAction(i, self.actionHistory[ts], self.channels))
            elif self.dm.algorithm == Algorithm.CNN:
                for i in range(len(self.sectors)):
                    actions.append(self.dm.takeAction(i, self.channels))
            """calculate capacity"""
            currentCapacity = calCapacity(actions, self.channels)
            """record"""
            self.actionHistory.append(actions)
            self.capacity.append(currentCapacity)
            average = sum(currentCapacity) / len(currentCapacity)
            self.averageCapacity.append(average)
            """update"""
            self.updateChannel()
            """print log"""
            self.logger.info(f'mode: {self.dm.algorithm}, time slot: {ts + 1}, system average capacity: {average}')


if __name__ == "__main__":
    mn = MobileNetwork()
    plotMobileNetwork(mn.getSectors(), mn.getUEs())
