from config import *
from channel_generator import generateChannel
from utils import generateChannelIndex, SKIP_LIST


class Environment:
    def __init__(self, sectors, UEs):
        self.channels = generateChannel(sectors, UEs)
        self.topPathLossList = self._calTopPathLoss_()

    def getChannel(self, transIndex, receiveIndex):
        return self.channels[generateChannelIndex(transIndex, receiveIndex)]

    def getTopPathLossList(self, i):
        """return a list of link index"""
        return self.topPathLossList[i]

    def update(self):
        for channel in self.channels.values():
            channel.update()

    def isIsolated(self, i, j):
        return j in SKIP_LIST[i]

    def isDirectLink(self, i, j):
        return i == j

    def _calTopPathLoss_(self):
        """top N j->i path loss"""
        topPathLossList = {}
        for i in range(3 * CELL_NUMBER):
            # get list of (link index, path loss)
            pathLoss = []
            for j in range(3 * CELL_NUMBER):
                if self.isDirectLink(i, j) or self.isIsolated(i, j):
                    continue
                else:
                    pathLoss.append((j, self.getChannel(j, i).getPathLoss()))
            # sort
            sortedPathLoss = sorted(pathLoss, key=lambda tup: tup[1])
            linkIndexes = [pathLoss[0] for pathLoss in sortedPathLoss[0:TOP_PATH_LOSS]]
            topPathLossList[i] = linkIndexes
        return topPathLossList
