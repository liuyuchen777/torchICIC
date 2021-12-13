import numpy as np


class Config:
    def __init__(self):
        # base station and user terminal
        self.BSAntenna = 4
        self.UTAntenna = 4
        self.BSHeight = 10.
        self.UTHeight = 1.5

        # power level list
        self.maxPower = 10.  # dBm
        self.powerLevel = 5
        self.powerList = []
        self._calPowerList_()

        # beamforming vector list
        self.codebookSize = 4
        self.beamformList = np.zeros(shape=[self.codebookSize, self.BSAntenna], dtype=np.cdouble)
        self._calCodeList_()

        # wireless channel
        self.alpha = 3  # path loss exponent
        self.ShadowingSigma = 8   # db
        self.gaussianSigma = 1
        # loss, non-loss
        self.noisePower = -100    # Gaussian white noise, dBm
        self.pathNumber = 6    # LOS path number

        # cellular network
        self.cellSize = 100.  # m
        self.cellNumber = 7
        self.Rmin = 15.
        self.Rmax = 60.
        self.interferencePenaltyAlpha = 0.2

        # memory pool
        self.mpMaxSize = 10000
        self.batchSize = 512

        # deep learning hyper-parameter
        self.totalTimeSlot = 1000000
        self.learningRate = 1e-4
        self.regBeta = 0.
        self.tStep = 256
        self.gamma = 0.3
        self.epsilon = 0.3
        self.evalTimes = 10
        self.hiddenLayers = [1024, 1024, 1024]
        self.inputLayer = 576
        self.outputLayer = 729

    def _calPowerList_(self):
        powerGap = self.maxPower * 2 / (self.powerLevel - 1)
        tmpPower = self.maxPower
        for i in range(self.powerLevel):
            self.powerList.append(tmpPower)
            tmpPower -= powerGap

    def _calCodeList_(self):
        # need to generate three sector of code matrix
        N = 16  # number of phases
        M = self.BSAntenna      # number of Antenna
        K = self.codebookSize   # codebook size
        for m in range(1, M+1):
            for k in range(1, K+1):
                self.beamformList[k-1][m-1] = np.exp(2j * np.pi / N * int(m * (k + K / 2) % K / (K/N))) / np.sqrt(M)


if __name__ == "__main__":
    config = Config()
    print(config.powerList)
    print(config.beamformList)
