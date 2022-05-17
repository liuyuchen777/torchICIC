from config import *


def calDistance(sectorPosition, uePosition):
    dis = np.power(sectorPosition[0] - uePosition[0], 2) \
          + np.power(sectorPosition[1] - uePosition[1], 2) \
          + np.power(sectorPosition[2] - uePosition[2], 2)

    return np.sqrt(dis)


class Channel:
    def __init__(self, sectorPosition, uePosition, ricianFactor=RICIAN_FACTOR):
        self.distance = calDistance(sectorPosition, uePosition)
        self.ricianFactor = ricianFactor
        self.beta = np.sqrt(1 / (np.power(self.distance / 1000, ALPHA)))
        self.CSI = self._calCSI_()
        self.CSIHistory = self.CSI

    def _calCSI_(self):
        """
        Calculate small-scale fading
        Returns:
            single time slot small-scale fading
        """
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
                thetaReceive = np.random.rand() * 2 * np.pi  # receive angle could be [0, 2pi)
            for m in range(UT_ANTENNA):
                AoA[m][0] = np.exp(-np.pi * np.sin(thetaReceive) * 1j * m)

            # h
            if path == 0:
                h = np.sqrt(self.ricianFactor / (1 + self.ricianFactor))
            else:
                hReal = np.random.normal(0., GAUSSIAN_SIGMA)
                hImage = np.random.normal(0., GAUSSIAN_SIGMA)
                h = hReal + 1j * hImage
                h = h * np.sqrt(1 / ((1 + self.ricianFactor) * (PATH_NUMBER - 1)))
            csi += h * AoA * np.transpose(AoD)
        return csi * self.beta

    def update(self):
        self.CSIHistory = self.CSI
        # self.CSI = rho * self.CSIHistory + np.sqrt(1 - rho ** 2) * self._calCSI_()
        self.CSI = self._calCSI_()

    def getCSI(self):
        return self.CSI

    def getCSIHistory(self):
        return self.CSIHistory

    def setRicianFactor(self, ricianFactor):
        self.ricianFactor = ricianFactor
