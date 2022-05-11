import numpy as np

from config import *
from utils import index2str, neighborTable, judgeSkip, dBm2num
from channel import Channel


class Environment:
    def __init__(self, CUs):
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
        for channel in self.channels.values():
            channel.step()

    """Partial Calculation Method"""

    def calDirectChannel(self, cu, sector, actionIndex):
        beamformer = BEAMFORMER_LIST[actionIndex[sector.index][0]]
        power = POWER_LIST[actionIndex[sector.index][1]]
        index = [cu.index, sector.index, cu.index, sector.index]

        directChannel = self.channels[index2str(index)].getCSI()
        tmpPower = dBm2num(power)
        tmpChannel = np.power(np.linalg.norm(np.matmul(directChannel, beamformer)), 4)
        signalPower = tmpPower * tmpChannel

        return signalPower

    def calIntraCellInterference(self, cu, sector, actionIndex):
        # direct channel
        directIndex = [cu.index, sector.index, cu.index, sector.index]
        directChannel = self.channels[index2str(directIndex)].getCSI()
        directBeamformer = BEAMFORMER_LIST[actionIndex[sector.index][0]]

        # intra-interference calculation
        intraCellInterference = 0.

        for otherSector in cu.sectors:
            if sector.index != otherSector.index:
                beamformer = BEAMFORMER_LIST[actionIndex[otherSector.index][0]]
                power = POWER_LIST[actionIndex[otherSector.index][1]]
                index = [cu.index, otherSector.index, cu.index, sector.index]
                # same cu channel
                intraChannel = self.channels[index2str(index)].getCSI()
                tmpPower = dBm2num(power)
                tmpChannel = np.power(np.linalg.norm(
                    np.matmul(
                        np.matmul(directBeamformer.transpose().conjugate(), directChannel.transpose().conjugate()),
                        np.matmul(intraChannel, beamformer)
                    )
                ), 2)
                intraCellInterference += tmpPower * tmpChannel

        return intraCellInterference

    def calNoisePower(self, cu, sector, actionIndex):
        beamformer = BEAMFORMER_LIST[actionIndex[sector.index][0]]
        index = [cu.index, sector.index, cu.index, sector.index]
        directChannel = self.channels[index2str(index)].getCSI()

        return dBm2num(NOISE_POWER) * np.power(np.linalg.norm(np.matmul(directChannel, beamformer)), 2)

    def calInterCellInterference(self, cu, sector, actionIndex):
        # direct channel
        directIndex = [cu.index, sector.index, cu.index, sector.index]
        directChannel = self.channels[index2str(directIndex)].getCSI()
        directBeamformer = BEAMFORMER_LIST[actionIndex[sector.index][0]]

        # inter-CU
        interCellInterference = 0.

        for otherCUIndex in neighborTable[cu.index]:
            otherCU = self.CUs[otherCUIndex]
            otherActionIndex = otherCU.getActionHistory()
            for otherCUSector in otherCU.sectors:
                index = [otherCU.index, otherCUSector.index, cu.index, sector.index]
                if judgeSkip(index):
                    continue
                beamformer = BEAMFORMER_LIST[otherActionIndex[otherCUSector.index][0]]
                power = POWER_LIST[otherActionIndex[otherCUSector.index][1]]
                ocsChannel = self.channels[index2str(index)].getCSI()
                tmpPower = dBm2num(power)
                tmpChannel = np.power(np.linalg.norm(
                    np.matmul(
                        np.matmul(directBeamformer.transpose().conjugate(), directChannel.transpose().conjugate()),
                        np.matmul(ocsChannel, beamformer)
                    )
                ), 2)
                interCellInterference += tmpPower * tmpChannel

        return interCellInterference

    """Integration Calculation Method"""

    def calLocalReward(self, CUIndex, actionIndex):
        """
        This reward only consider Intra-CU interference for single CU in specific action Index
        This reward calculation function is used by Cell ES algorithm
        """
        cu = self.CUs[CUIndex]
        r = 0.
        for sector in cu.sectors:
            index = [cu.index, sector.index, cu.index, sector.index]
            signalPower = self.calDirectChannel(cu, sector, actionIndex)
            noisePower = self.calNoisePower(cu, sector, actionIndex)
            intraCellInterference = self.calIntraCellInterference(cu, sector, actionIndex)
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
            actionIndex = cu.getAction()
            r = 0.
            for sector in cu.sectors:
                signalPower = self.calDirectChannel(cu, sector, actionIndex)
                noisePower = self.calNoisePower(cu, sector, actionIndex)
                intraCellInterference = self.calIntraCellInterference(cu, sector, actionIndex)
                interCellInterference = self.calInterCellInterference(cu, sector, actionIndex)

                SINR = signalPower / (noisePower + interCellInterference + intraCellInterference)
                cap = np.log2(1 + SINR)
                r += cap
            r /= 3
            reward.append(r)
        # interference penalty
        rewardRevised = []
        alpha = INFERENCE_PENALTY_ALPHA
        for CUIndex in range(len(reward)):
            extraReward = 0.
            for neighborCU in neighborTable[CUIndex]:
                extraReward += reward[neighborCU]
            extraReward /= len(neighborTable[CUIndex])
            tmpReward = (1 - alpha) * reward[CUIndex] + alpha * extraReward
            rewardRevised.append(tmpReward)

        return rewardRevised


if __name__ == "__main__":
    print("----------------Test ENV----------------")
