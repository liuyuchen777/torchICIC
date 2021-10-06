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
        np.random.seed(seed=None)
        self.logger = logging.getLogger(__name__)

    def _calDistance_(self):
        dis = (self.sector.pos[0] - self.ue.pos[0]) ** 2 \
              + (self.sector.pos[1] - self.ue.pos[1]) ** 2 \
              + (self.sector.pos[2] - self.ue.pos[2]) ** 2

        return np.sqrt(dis)

    def _calCSI_(self):
        index = self.index[1]
        # large scale
        beta = dB2num(120.9 + 37.6 * np.log10(self.distance / 1000) + np.random.normal(0, self.config.logNormalSigma))
        # empty csi
        csi = np.zeros(shape=[self.config.UTAntenna, self.config.BSAntenna], dtype=complex)
        for i in range(self.config.pathNumber):
            # Angle of Arrival
            AoA = np.zeros(shape=[1, self.config.UTAntenna], dtype=complex)
            # Angle of Departure
            AoD = np.zeros(shape=[1, self.config.BSAntenna], dtype=complex)
            # Average Distribution
            theta_s = (np.random.rand() * 120 + 120 * index) / 360 * 2 * np.pi
            for n in range(self.config.BSAntenna):
                AoD[0][n] = np.exp(-np.pi*self.distance*np.sin(theta_s)*1j*(n-1))
            theta_r = np.random.rand() * 2 * np.pi
            for m in range(self.config.UTAntenna):
                AoA[0][m] = np.exp(-np.pi*self.distance*np.sin(theta_r)*1j*(m-1))
            # complex Gaussian random variable
            h = np.random.normal(loc=0., scale=dB2num(self.config.gaussianSigma), size=(1, 2)).view(dtype=complex)
            # print("h: ", h)
            csi += h * AoA * np.transpose(AoD)
        csi /= np.sqrt(beta * self.config.pathNumber)
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
    # these are the values over wich your kernel will be evaluated
    dist_space = linspace(min(data), max(data), 100)
    # plot the results
    plt.plot(dist_space, kde(dist_space))
    plt.show()


if __name__ == "__main__":
    mSector = Sector(0, 0, [0., 0., 100.])
    mUE = UE(0, 0, [100., 0., 10.])
    channels = []
    for i in range(1000):
        channel = Channel(sector=mSector, ue=mUE)
        norm = np.linalg.norm(channel.getCSI())
        channels.append(norm)
    plotPDF(channels)
