import random

from config import *
from utils import dB2num


def calDistance(sectorPosition, uePosition):
    dis = np.power(sectorPosition[0] - uePosition[0], 2) \
          + np.power(sectorPosition[1] - uePosition[1], 2) \
          + np.power(sectorPosition[2] - uePosition[2], 2)

    return np.sqrt(dis)


class Channel:
    def __init__(self, sectorPosition, uePosition, ricianFactor=RICIAN_FACTOR):
        self.distance = calDistance(sectorPosition, uePosition)
        self.ricianFactor = dB2num(ricianFactor)
        self.pathLoss = self._calPathLoss_()
        self.CSI = self._calCSI_()

    def _calPathLoss_(self):
        shadowing = dB2num(SHADOWING_SIGMA * random.random())
        return 1 / np.sqrt(self.distance ** ALPHA)

    def _calCSI_(self):
        """
        Calculate small-scale fading
        Returns:
            single time slot small-scale fading
        """
        csi = np.zeros(shape=[UT_ANTENNA, BS_ANTENNA], dtype=complex)
        for path in range(PATH_NUMBER):
            # AoA and AoD
            AoD = np.zeros(shape=[BS_ANTENNA, 1], dtype=complex)
            AoA = np.zeros(shape=[UT_ANTENNA, 1], dtype=complex)
            thetaSend = np.random.rand() * 2 * np.pi
            thetaReceive = np.random.rand() * 2 * np.pi
            for n in range(BS_ANTENNA):
                AoD[n][0] = np.exp(-np.pi * np.sin(thetaSend) * 1j * n)
            for m in range(UT_ANTENNA):
                AoA[m][0] = np.exp(-np.pi * np.sin(thetaReceive) * 1j * m)

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

