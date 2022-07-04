import random

import matplotlib.pyplot as plt

from config import *
from utils import dB2num, pdf


def plotRicianChannel():
    """[test] pdf of channel"""
    channel = Channel([0., 0., 10.], [100., 100., 1.5])

    channel.setRicianFactor(10)
    channels = []
    for i in range(5000):
        norm = np.linalg.norm(channel.getCSI())
        channels.append(norm)
        channel.update()
    pdf(channels, linestyle=":", label="K=10")

    channel.setRicianFactor(5)
    channels = []
    for i in range(5000):
        norm = np.linalg.norm(channel.getCSI())
        channels.append(norm)
        channel.update()
    pdf(channels, linestyle="-", label="K=5")

    channel.setRicianFactor(2)
    channels = []
    for i in range(5000):
        norm = np.linalg.norm(channel.getCSI())
        channels.append(norm)
        channel.update()
    pdf(channels, linestyle="--", label="K=2")

    plt.legend(loc='upper right')
    plt.xlabel("Norm of Channel")
    plt.ylabel("Probability")
    plt.show()


def calDistance(sectorPosition, uePosition):
    dis = np.power(sectorPosition[0] - uePosition[0], 2) \
          + np.power(sectorPosition[1] - uePosition[1], 2) \
          + np.power(sectorPosition[2] - uePosition[2], 2)

    return np.sqrt(dis)


class Channel:
    def __init__(self, sectorPosition, uePosition, ricianFactor=RICIAN_FACTOR, isShadowing=True):
        self.distance = calDistance(sectorPosition, uePosition)
        self.ricianFactor = dB2num(ricianFactor)
        self.pathLoss = self._calPathLoss_(isShadowing)
        self.CSI = self._calCSI_()

    def _calPathLoss_(self, isShadowing):
        if isShadowing:
            shadowing = dB2num(SHADOWING_SIGMA * random.random())
            return 1 / np.sqrt(self.distance ** ALPHA + shadowing)
        else:
            return 1 / np.sqrt(self.distance ** ALPHA)

    def _calAoAAoD_(self):
        AoD = np.zeros(shape=[BS_ANTENNA, 1], dtype=complex)
        AoA = np.zeros(shape=[UT_ANTENNA, 1], dtype=complex)
        thetaSend = np.random.rand() * 2 * np.pi
        thetaReceive = np.random.rand() * 2 * np.pi
        for n in range(BS_ANTENNA):
            AoD[n][0] = np.exp(-np.pi * np.sin(thetaSend) * 1j * n)
        for m in range(UT_ANTENNA):
            AoA[m][0] = np.exp(-np.pi * np.sin(thetaReceive) * 1j * m)
        return AoA, AoD

    def _calCSI_(self):
        """
        Calculate small-scale fading
        Returns:
            single time slot small-scale fading
        """
        csi = np.zeros(shape=[UT_ANTENNA, BS_ANTENNA], dtype=complex)
        for path in range(PATH_NUMBER):
            AoA, AoD = self._calAoAAoD_()
            # h
            if path == 0:
                h = np.sqrt(self.ricianFactor / (1 + self.ricianFactor))    # LoS
            else:
                hTheta = np.random.rand() * 2 * np.pi
                h = np.cos(hTheta) + 1j * np.sin(hTheta)
                h = h * np.sqrt(1 / ((1 + self.ricianFactor) * (PATH_NUMBER - 1)))
            csi += h * np.matmul(AoA, np.transpose(AoD))
        csi = csi * self.pathLoss
        return csi

    def update(self):
        self.CSI = self._calCSI_()

    def getCSI(self):
        return self.CSI

    def setRicianFactor(self, ricianFactor):
        self.ricianFactor = ricianFactor

    def getPathLoss(self):
        return self.pathLoss


if __name__ == "__main__":
    EXECUTION_MODE = "PRINT_SINGLE_CHANNEL_CSI"
    if EXECUTION_MODE == "PRINT_SINGLE_CHANNEL_CSI":
        channel = Channel([0., 0., 10.], [10., 10., 1.5])
        print("CSI: ")
        print(channel.getCSI())
    elif EXECUTION_MODE == "PLOT_CHANNEL_PDF":
        plotRicianChannel()
    elif EXECUTION_MODE == "TEST_PLOT_PDF":
        gaussianData = np.random.normal(loc=0., scale=1., size=10000)
        pdf(gaussianData)
        plt.hist(gaussianData, color='blue', edgecolor='black', bins=2000)
        plt.show()
