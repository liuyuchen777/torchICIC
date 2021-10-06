import numpy as np


class Config:
    def __init__(self):
        # base station and user terminal
        self.BSAntenna = 4
        self.UTAntenna = 4
        self.BSHeight = 10.
        self.UTHeight = 1.5

        # power level list
        self.maxPower = 10.
        self.powerLevel = 5
        self.powerList = []
        self._calPowerList_()

        # beamforming vector list
        self.codebookSize = 4
        self.beamformList = np.zeros(shape=[self.BSAntenna, self.codebookSize], dtype=np.cdouble)
        self._calCodeList_()

        # wireless channel
        self.alpha = 4  # path loss exponent
        self.logNormalSigma = 8   # db
        self.gaussianSigma = 10    # db
        self.noisePower = -100    # Gaussian white noise, dBm
        self.pathNumber = 6    # LOS path number

        # cellular network
        self.cellLength = 200. # m
        self.cellNumber = 7
        self.Rmin = 40.
        self.Rmax = 100.

        # memory pool
        self.mpMaxSize = 10000
        self.batchSize = 1000

        # deep learning hyper-parameter
        self.totalTimeSlot = 1000

    def _calPowerList_(self):
        powerGap = self.maxPower * 2 / (self.powerLevel - 1)
        tmpPower = self.maxPower
        for i in range(self.powerLevel):
            self.powerList.append(tmpPower)
            tmpPower -= powerGap

    def _calCodeList_(self):
        # need to generate three sector of code matrix
        S = 16
        N = self.BSAntenna
        Q = self.codebookSize
        for n in range(N):
            for q in range(Q):
                self.beamformList[q][n] = np.exp(1j * 2 * np.pi / S * (n * ((q + Q / 2) % Q) / (Q / S))) / np.sqrt(N)


if __name__ == "__main__":
    config = Config()
    print(config.powerList)
    print(config.beamformList)
