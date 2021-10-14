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
        self.channels = dict()  # dict(channel_index) -> Channel, channel index is string
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

    def getChannels(self):
        return self.channels

    def getChannel(self, index):
        """index should be string"""
        return self.channels[index]

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
                        beamformer = self.config.beamformList[actionIndex[otherSector.index][0]]
                        # same cu channel
                        scChannel = self.channels[index2str([cu.index, otherSector.index, cu.index, sector.index])] \
                            .getCSI()
                        bottom += dBm2num(self.config.powerList[actionIndex[otherSector.index][1]]) \
                            * np.linalg.norm(np.matmul(np.matmul(np.matmul(np.transpose(beamformer),
                            np.transpose(scChannel)), scChannel), beamformer)) ** 2
                # 2.3 inter-CU interference
                for otherCUIndex in neighborTable[cu.index]:
                    otherCU = self.CUs[otherCUIndex]
                    otherActionIndex = otherCU.getDecisionIndex()
                    for otherCUSector in otherCU.sectors:
                        index = [cu.index, sector.index, otherCU.index, otherCUSector.index]
                        if judgeSkip(index):
                            # if inter-CU sector can't interfere current sector, skip!
                            continue
                        beamformer = self.config.beamformList[otherActionIndex[otherCUSector.index][0]]
                        ocsChannel = self.channels[index2str(index)].getCSI()
                        bottom += dBm2num(self.config.powerList[otherActionIndex[otherCUSector.index][1]]) * \
                            np.linalg.norm(np.matmul(np.matmul(np.matmul(np.transpose(beamformer),
                            np.transpose(ocsChannel)), ocsChannel), beamformer)) ** 2
                # 3. use SINR calculate capacity
                SINR = up / bottom
                cap = np.log2(1 + SINR)
                r += cap
            # 4. calculate average reward in CU
            r /= 3
            # 5. multiply interference penalty
            # 5.1 calculate
            interferencePenalty = 0.
            for sector in range(3):
                for otherCUIndex in neighborTable[cu.index]:
                    for otherSector in range(3):
                        index = [otherCUIndex, otherSector, cu.index, sector]
                        if judgeSkip(index):
                            continue
                        else:
                            # 4.1 get channel from channel dict
                            power = self.config.powerList[actionIndex[sector][1]]
                            beamformer = self.config.beamformList[actionIndex[sector][0]]
                            channel = self.channels[index2str(index)].getCSI()
                            interferencePenalty += dBm2num(power) * \
                                np.linalg.norm(np.matmul(np.matmul(np.matmul(np.transpose(beamformer),
                                np.transpose(channel)), channel), beamformer))
            # 5.2 multiply
            r = r * (1 - self.config.interferencePenaltyAlpha * interferencePenalty)
            reward.append(r)

        return reward
