import logging
from Config import Config
from utils import dB2num, setLogger
import numpy as np
from Sector import Sector
from UserEquipment import UE
import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde
from numpy import linspace

"""
bs_antenna = N
ue_antenna = M
CSI is M * N matrix
"""


class Channel:
    def __init__(self, sector, ue):
        self.config = Config()
        self.sector = sector
        self.ue = ue
        self.index = [sector.CUIndex, sector.index, ue.CUIndex, ue.index]  # (CU of sector, sector, CU of ue, ue)
        self.distance = self._calDistance_()
        self.CSI = self._calCSI_()
        self.CSIHistory = self.CSI     # csi in last time slot

    def _calDistance_(self):
        dis = (self.sector.pos[0] - self.ue.pos[0]) ** 2 \
              + (self.sector.pos[1] - self.ue.pos[1]) ** 2 \
              + (self.sector.pos[2] - self.ue.pos[2]) ** 2

        return np.sqrt(dis)

    def _calCSI_(self):
        index = self.index[1]
        # large-scale fading
        beta = dB2num(120.9 + 37.6 * np.log10(self.distance / 1000) + np.random.normal(0, self.config.logNormalSigma))
        # empty csi
        csi = np.zeros(shape=[self.config.UTAntenna, self.config.BSAntenna], dtype=complex)
        for _ in range(self.config.pathNumber):
            # Angle of Arrival
            AoA = np.zeros(shape=[self.config.UTAntenna, 1], dtype=complex)
            # Angle of Departure
            AoD = np.zeros(shape=[self.config.BSAntenna, 1], dtype=complex)
            """
            sender angle:
            index 0 -> [0, 120)
            index 1 -> [120, 240)
            index 2 -> [240, 360)
            """
            theta_s = (np.random.rand() * 120 + 120 * index) / 360 * 2 * np.pi
            for n in range(self.config.BSAntenna):
                AoD[n][0] = np.exp(-np.pi*self.distance*np.sin(theta_s)*1j*n)
            theta_r = np.random.rand() * 2 * np.pi      # receive angle could be [0, 2pi)
            for m in range(self.config.UTAntenna):
                AoA[m][0] = np.exp(-np.pi*self.distance*np.sin(theta_r)*1j*m)
            # complex Gaussian random variable
            h = np.random.normal(loc=0., scale=dB2num(self.config.gaussianSigma), size=(1, 2)).view(dtype=complex)
            # print("h: \n", h)
            # print("AoD: \n", AoD)
            # print("AoA: \n", AoA)
            # print("AoA * AoD: \n", AoA * np.transpose(AoD))
            csi += h * AoA * np.transpose(AoD)
        csi /= np.sqrt(beta)
        return csi

    def step(self):
        self.CSIHistory = self.CSI
        self.CSI = self._calCSI_()

    def getCSI(self):
        return self.CSI

    def getCSIHistory(self):
        return self.CSIHistory


def plotPDF(data):
    # this create the kernel, given an array it will estimate the probability over that values
    kde = gaussian_kde(data)
    # these are the values over which your kernel will be evaluated
    distSpace = linspace(min(data), max(data), 1000)
    # plot the results
    plt.plot(distSpace, kde(distSpace))
    plt.show()


if __name__ == "__main__":
    """[test] single channel value"""
    # mSector = Sector(0, 0, [0., 0., 10.])
    # mUE = UE(0, 0, [50., 50., 1.5])
    # channel = Channel(sector=mSector, ue=mUE)
    # print("CSI: ")
    # print(channel.getCSI())
    """[test] pdf of channel"""
    mSector = Sector(0, 0, [0., 0., 10.])
    mUE = UE(0, 0, [50., 50., 1.5])
    channel = Channel(sector=mSector, ue=mUE)
    channels = []
    for i in range(10000):
        norm = np.linalg.norm(channel.getCSI())
        channels.append(norm)
        channel.step()
    plotPDF(channels)
