from config import *
from utils import dB2num
import numpy as np
from sector import Sector
from user_equipment import UE
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
        self.sector = sector
        self.ue = ue
        self.index = [sector.CUIndex, sector.index, ue.CUIndex, ue.index]
        self.distance = self._calDistance_()
        self.shadowing = np.random.normal(0., SHADOWING_SIGMA)
        self.largeScale = np.power(self.distance / 1000, ALPHA)
        self.beta = 1 / dB2num(self.largeScale)
        self.CSI = self._calCSI_()
        self.CSIHistory = self.CSI     # csi in last time slot

    def _calDistance_(self):
        dis = np.power(self.sector.pos[0] - self.ue.pos[0], 2) \
              + np.power(self.sector.pos[1] - self.ue.pos[1], 2) \
              + np.power(self.sector.pos[2] - self.ue.pos[2], 2)

        return np.sqrt(dis)

    def _calCSI_(self):
        # empty csi
        csi = np.zeros(shape=[UT_ANTENNA, BS_ANTENNA], dtype=complex)
        for path in range(PATH_NUMBER):
            # Angle of Departure
            AoD = np.zeros(shape=[BS_ANTENNA, 1], dtype=complex)
            if path == 0:
                thetaSend = 0
            else:
                thetaSend = np.random.rand() * 2 * np.pi
            for n in range(BS_ANTENNA):
                AoD[n][0] = np.exp(-np.pi * np.sin(thetaSend) * 1j * n)

            # Angle of Arrival
            AoA = np.zeros(shape=[UT_ANTENNA, 1], dtype=complex)
            if path == 0:
                thetaReceive = 0
            else:
                thetaReceive = np.random.rand() * 2 * np.pi      # receive angle could be [0, 2pi)
            for m in range(UT_ANTENNA):
                AoA[m][0] = np.exp(-np.pi * np.sin(thetaReceive) * 1j * m)

            # h
            if path == 0:
                h = np.sqrt(RICIAN_FACTOR / (1 + RICIAN_FACTOR))
            else:
                hReal = np.random.normal(0., GAUSSIAN_SIGMA)
                hImage = np.random.normal(0., GAUSSIAN_SIGMA)
                h = hReal + 1j * hImage
                h = h * np.sqrt(1 / ((1 + RICIAN_FACTOR) * (PATH_NUMBER - 1)))
            csi += h * AoA * np.transpose(AoD)
        csi = csi * np.sqrt(self.beta)
        return csi

    def step(self):
        self.CSIHistory = self.CSI
        # self.CSI = rho * self.CSIHistory + np.sqrt(1 - rho ** 2) * self._calCSI_()
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


if __name__ == "__main__":
    EXECUTION_MODE = "CHANNEL_PDF"

    if EXECUTION_MODE == "SINGLE_CHANNEL":
        """[test] single channel value"""
        mSector = Sector(0, 0, [0., 0., 10.])
        mUE = UE(0, 0, [10., 10., 1.5])
        channel = Channel(sector=mSector, ue=mUE)
        print("CSI: ")
        print(channel.getCSI())
    elif EXECUTION_MODE == "CHANNEL_PDF":
        """[test] pdf of channel"""
        mSector = Sector(0, 0, [0., 0., 10.])
        mUE = UE(0, 0, [10., 10., 1.5])

        RICIAN_FACTOR = 10
        channel = Channel(sector=mSector, ue=mUE)
        channels = []
        for i in range(20000):
            norm = np.linalg.norm(channel.getCSI())
            channels.append(norm)
            channel.step()
        plotPDF(channels)

        RICIAN_FACTOR = 5
        channels = []
        for i in range(20000):
            norm = np.linalg.norm(channel.getCSI())
            channels.append(norm)
            channel.step()
        plotPDF(channels)

        RICIAN_FACTOR = 2
        channels = []
        for i in range(20000):
            norm = np.linalg.norm(channel.getCSI())
            channels.append(norm)
            channel.step()
        plotPDF(channels)

        plt.show()
    elif EXECUTION_MODE == "TEST_PLOT_PDF":
        """[test] pdf function work or not"""
        gaussianData = np.random.normal(loc=0., scale=1., size=100000)
        plotPDF(gaussianData)
        plt.hist(gaussianData, color='blue', edgecolor='black', bins=2000)
        plt.show()
