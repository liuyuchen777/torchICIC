import logging
from config import Config
from utils import dB2num, set_logger
import numpy as np
from sector import Sector
from user_equipment import UE

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
        self.index = [sector.CU_index, sector.index, ue.CU_index, ue.index]  # (CU of sector, sector, CU of ue, ue)
        self.distance = self._cal_distance_()
        self.csi = self._cal_csi_()
        self.csi_history = self.csi
        np.random.seed(seed=None)
        self.logger = logging.getLogger(__name__)

    def _cal_distance_(self):
        dis = (self.sector.pos[0] - self.ue.pos[0]) ** 2 \
              + (self.sector.pos[1] - self.ue.pos[1]) ** 2 \
              + (self.sector.pos[2] - self.ue.pos[2]) ** 2

        return np.sqrt(dis)

    def _cal_csi_(self):
        index = self.index[1]
        # large scale
        beta = 1 / dB2num(120.9 + 37.6 * np.log10(self.distance / 1000)
                          + np.random.normal(0, self.config.log_normal_sigma))
        # empty csi
        csi = np.zeros(shape=[self.config.UT_antenna, self.config.BS_antenna], dtype=complex)
        for i in range(self.config.path_number):
            # Angle of Arrival
            AoA = np.zeros(shape=[1, self.config.UT_antenna], dtype=complex)
            # Angle of Departure
            AoD = np.zeros(shape=[1, self.config.BS_antenna], dtype=complex)
            # Average Distribution
            theta = (np.random.rand() * 120 + 240 + 120 * index) / 360 * 2 * np.pi
            for n in range(self.config.BS_antenna):
                AoD[0][n] = np.exp(-2*np.pi*self.distance*np.cos(theta)/self.config.wave_length*1j*(n-1))
            for m in range(self.config.UT_antenna):
                AoA[0][m] = np.exp(-2*np.pi*self.distance*np.cos(theta)/self.config.wave_length*1j*(m-1))
            # complex Gaussian random variable
            h = np.random.normal(loc=0., scale=dB2num(self.config.Gaussian_sigma), size=(1, 2)).view(dtype=complex)
            # print("h: ", h)
            csi += h * AoA * np.transpose(AoD)
        csi *= beta
        return csi

    def step(self):
        self.csi_history = self.csi
        self.csi = self._cal_csi_()

    def get_csi(self):
        return self.csi

    def get_csi_history(self):
        return self.csi_history


if __name__ == "__main__":
    set_logger()
    logger = logging.getLogger(__name__)
    channel = Channel(sector=Sector(0, 0, [0., 0., 100.]), ue=UE(0, 0, [100., 0., 10.]))
    print("distance: ", channel.distance)
