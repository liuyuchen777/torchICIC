import logging
import numpy as np
from Config import Config
from utils import index2str, neighborTable, judgeSkip, dBm2num
from Channel import Channel


class Environment:
    def __init__(self, CUs):
        self.logger = logging.getLogger(__name__)
        self.config = Config()
        self.CUs = CUs
        self.channels = dict()  # dict(channel_index) -> Channel
        self._initChannel_()

    def _initChannel_(self):
        """
        build all channels
        """
        for cu in self.CUs:
            for sector in cu.sectors:
                # intra CU
                for ue in cu.UEs:
                    channel = Channel(sector, ue)
                    self.channels[index2str(channel.index)] = channel
                # inter CU
                for otherCU in self.CUs:
                    if otherCU.index == cu.index:
                        continue
                    for ue in otherCU.UEs:
                        channel = Channel(sector, ue)
                        self.channels[index2str(channel.index)] = channel

    def getChannel(self):
        return self.channels

    def printChannel(self):
        for (k, v) in self.channels.items():
            print(f'CSI of index {k}:')
            print(f'{v.getCSI()}')

    def step(self):
        for v in self.channels.values():
            v.step()

    def calReward(self):
        """
        use action and CSI calculate each CU's reward
        return in list, length of CU number
        NOTICE: characteristic of three-sector model
        """
        reward = []
        for cu in self.CUs:
            # main loop calculate reward in each CU
            actionIndex = cu.getDecisionIndex()
            r = 0.
            for sector in cu.sectors:
                w_i_k = self.config.beamformList[actionIndex[sector.index][0]]
                # 1. direct channel (up)
                direct_channel = self.channels[index2str([cu.index, sector.index, cu.index, sector.index])].getCSI()
                w_i_k = np.matmul(direct_channel, w_i_k)
                up = dBm2num(self.config.powerList[actionIndex[sector.index][1]]) * np.linalg.norm(w_i_k) ** 4
                # 2.1 Gaussian noise
                bottom = dBm2num(self.config.noisePower) * np.linalg.norm(w_i_k) ** 2
                # 2.2 intra-CU interference
                for otherSector in cu.sectors:
                    if sector.index != otherSector.index:
                        w_j_k = self.config.beamformList[actionIndex[otherSector.index][0]]
                        # same cu channel
                        scChannel = self.channels[index2str([cu.index, otherSector.index, cu.index, sector.index])] \
                            .getCSI()
                        bottom += dBm2num(self.config.powerList[actionIndex[otherSector.index][1]]) \
                            * np.linalg.norm(np.matmul(np.matmul(np.matmul(np.transpose(w_j_k),
                            np.transpose(scChannel)), scChannel), w_j_k)) ** 2
                # 2.3 inter-CU interference
                neighbor = neighborTable[cu.index]
                for otherCUIndex in neighbor:
                    otherCU = self.CUs[otherCUIndex]
                    for otherCUSector in otherCU.sectors:
                        index = [cu.index, sector.index, otherCU.index, otherCUSector.index]
                        if judgeSkip(index):
                            # if inter-CU sector can't interfere current sector, skip!
                            continue
                        w_j_l = self.config.beamformList[actionIndex[otherCUSector.index][0]]
                        ocsChannel = self.channels[index2str(index)].getCSI()
                        bottom += dBm2num(self.config.powerList[actionIndex[otherCUSector.index][1]]) * \
                                  np.linalg.norm(np.matmul(np.matmul(np.matmul(np.transpose(w_j_l),
                            np.transpose(ocsChannel)), ocsChannel), w_j_l)) ** 2
                # 3. use SINR calculate capacity
                SINR = up / bottom
                cap = np.log2(1 + SINR)
                r += cap
            # 4. calculate average reward in CU
            r /= 3
            reward.append(r)

        return reward
