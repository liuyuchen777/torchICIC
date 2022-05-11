import matplotlib.pyplot as plt
from scipy.stats.kde import gaussian_kde

from config import *


def calDistance(sectorPosition, uePosition):
    dis = np.power(sectorPosition[0] - uePosition[0], 2) \
          + np.power(sectorPosition[1] - uePosition[1], 2) \
          + np.power(sectorPosition[2] - uePosition[2], 2)

    return np.sqrt(dis)


def plotPDF(data):
    # create kernel, given an array it will estimate the probability over that values
    kde = gaussian_kde(data)
    # these are the values over which your kernel will be evaluated
    distSpace = np.linspace(min(data), max(data), 1000)
    # plot the results
    plt.plot(distSpace, kde(distSpace))


class Channel:
    def __init__(self, sectorPosition, uePosition):
        self.distance = calDistance(sectorPosition, uePosition)
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
                h = np.sqrt(RICIAN_FACTOR / (1 + RICIAN_FACTOR))
            else:
                hReal = np.random.normal(0., GAUSSIAN_SIGMA)
                hImage = np.random.normal(0., GAUSSIAN_SIGMA)
                h = hReal + 1j * hImage
                h = h * np.sqrt(1 / ((1 + RICIAN_FACTOR) * (PATH_NUMBER - 1)))
            csi += h * AoA * np.transpose(AoD)
        return csi * self.beta

    def update(self):
        self.CSIHistory = self.CSI
        self.CSI = rho * self.CSIHistory + np.sqrt(1 - rho ** 2) * self._calCSI_()
        # self.CSI = self._calCSI_()

    def getCSI(self):
        return self.CSI

    def getCSIHistory(self):
        return self.CSIHistory


if __name__ == "__main__":
    EXECUTION_MODE = "CHANNEL_PDF"

    if EXECUTION_MODE == "SINGLE_CHANNEL":
        """[test] single channel value"""
        channel = Channel([0., 0., 10.], [10., 10., 1.5])
        print("CSI: ")
        print(channel.getCSI())
    elif EXECUTION_MODE == "CHANNEL_PDF":
        """[test] pdf of channel"""
        channel = Channel([0., 0., 10.], [100., 100., 1.5])

        RICIAN_FACTOR = 10
        channels = []
        for i in range(20000):
            norm = np.linalg.norm(channel.getCSI())
            channels.append(norm)
            channel.update()
        plotPDF(channels)

        RICIAN_FACTOR = 5
        channels = []
        for i in range(20000):
            norm = np.linalg.norm(channel.getCSI())
            channels.append(norm)
            channel.update()
        plotPDF(channels)

        RICIAN_FACTOR = 2
        channels = []
        for i in range(20000):
            norm = np.linalg.norm(channel.getCSI())
            channels.append(norm)
            channel.update()
        plotPDF(channels)

        plt.show()
    elif EXECUTION_MODE == "TEST_PLOT_PDF":
        """[test] pdf function work or not"""
        gaussianData = np.random.normal(loc=0., scale=1., size=100000)
        plotPDF(gaussianData)
        plt.hist(gaussianData, color='blue', edgecolor='black', bins=2000)
        plt.show()
