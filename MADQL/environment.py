import logging
import numpy as np
from config import Config
from utils import index2str, neighborTable, judgeSkip, dBm2num
from channel import Channel


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
                # intra CU + direct channel
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

    def calLocalReward(self, CUIndex, actionIndex):
        """
        this reward only consider Intra-CU interference
        for single CU in specific action Index
        """
        cu = self.CUs[CUIndex]
        r = 0.
        for sector in cu.sectors:
            # 1. direct channel (up)
            beamformer = self.config.beamformList[actionIndex[sector.index][0]]
            power = self.config.powerList[actionIndex[sector.index][1]]
            index = [cu.index, sector.index, cu.index, sector.index]
            # direct channel
            directChannel = self.channels[index2str(index)].getCSI()
            signalPower = dBm2num(power) * np.power(np.linalg.norm(beamformer * directChannel), 4)
            # 2. bottom
            # 2.1 Gaussian noise
            noisePower = dBm2num(self.config.noisePower) * np.power(np.linalg.norm(beamformer * directChannel), 2)
            intraCellInterference = 0.
            # 2.2 intra-CU interference
            for otherSector in cu.sectors:
                if sector.index != otherSector.index:
                    beamformer = self.config.beamformList[actionIndex[otherSector.index][0]]
                    power = self.config.powerList[actionIndex[otherSector.index][1]]
                    index = [cu.index, otherSector.index, cu.index, sector.index]
                    # same cu channel
                    intraChannel = self.channels[index2str(index)].getCSI()
                    intraCellInterference += dBm2num(power) * np.power(np.linalg.norm(np.transpose(beamformer)
                                        * np.transpose(intraChannel) * intraChannel * beamformer), 2)
            # summation
            SINR = signalPower / (noisePower + intraCellInterference)
            cap = np.log2(1 + SINR)
            r += cap
        # 4. calculate average reward in CU
        r /= 3

        return r

    def calReward(self):
        """
        use action and CSI calculate each CU's reward
        return in list, length of CU number
        NOTICE: characteristic of three-sector model
        """
        reward = []
        for cu in self.CUs:
            # main loop calculate reward in each CU
            actionIndex = cu.getAction()
            r = 0.
            for sector in cu.sectors:
                # 1. direct channel
                beamformer = self.config.beamformList[actionIndex[sector.index][0]]
                power = self.config.powerList[actionIndex[sector.index][1]]
                index = [cu.index, sector.index, cu.index, sector.index]
                directChannel = self.channels[index2str(index)].getCSI()
                signalPower = dBm2num(power) * np.power(np.linalg.norm(beamformer * directChannel), 4)
                # 2. bottom
                # 2.1 Gaussian noise
                noisePower = dBm2num(self.config.noisePower) * np.power(np.linalg.norm(beamformer * directChannel), 2)
                intraCellInterference = 0.
                # 2.2 intra-CU interference
                for otherSector in cu.sectors:
                    if sector.index != otherSector.index:
                        beamformer = self.config.beamformList[actionIndex[otherSector.index][0]]
                        power = self.config.powerList[actionIndex[otherSector.index][1]]
                        index = [cu.index, otherSector.index, cu.index, sector.index]
                        # same cu channel
                        intraChannel = self.channels[index2str(index)].getCSI()
                        intraCellInterference += dBm2num(power) * np.power(np.linalg.norm(np.transpose(beamformer)
                                            * np.transpose(intraChannel) * intraChannel * beamformer), 2)
                # 2.3 inter-CU interference
                interCellInterference = 0.
                for otherCUIndex in neighborTable[cu.index]:
                    otherCU = self.CUs[otherCUIndex]
                    otherActionIndex = otherCU.getActionHistory()
                    for otherCUSector in otherCU.sectors:
                        index = [otherCU.index, otherCUSector.index, cu.index, sector.index]
                        if judgeSkip(index):
                            continue
                        beamformer = self.config.beamformList[otherActionIndex[otherCUSector.index][0]]
                        power = self.config.powerList[otherActionIndex[otherCUSector.index][1]]
                        ocsChannel = self.channels[index2str(index)].getCSI()
                        interCellInterference += dBm2num(power) * np.power(np.linalg.norm(np.transpose(beamformer)
                                            * np.transpose(ocsChannel) * ocsChannel * beamformer), 2)
                # 3. use SINR calculate capacity
                SINR = signalPower / (noisePower + interCellInterference + intraCellInterference)
                cap = np.log2(1 + SINR)
                r += cap
            # 4. calculate average reward in CU
            r /= 3
            reward.append(r)
        # 5. consider others
        rewardRevised = []
        alpha = self.config.interferencePenaltyAlpha
        for CUIndex in range(len(reward)):
            extraReward = 0.
            for neighborCU in neighborTable[CUIndex]:
                extraReward += reward[neighborCU]
            extraReward /= len(neighborTable[CUIndex])
            tmpReward = (1 - alpha) * reward[CUIndex] + alpha * extraReward
            rewardRevised.append(tmpReward)

        return rewardRevised
