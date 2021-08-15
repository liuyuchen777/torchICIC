import numpy as np


class Config:
    def __init__(self):
        # base station and user terminal
        self.BS_antenna = 4
        self.UT_antenna = 4
        self.BS_height = 10.
        self.UT_height = 1.5

        # power level list
        self.max_power = 10.
        self.power_level = 5
        self.power_list = []
        self._cal_power_list_()

        # beamforming vector list
        self.codebook_size = 4
        self.beamform_list = np.zeros(shape=[self.BS_antenna, self.codebook_size], dtype=np.cdouble)
        self._cal_code_list_()

        # channel
        self.Rician_factor = 10
        self.LOS_path = 6
        self.all_component = 8
        self.max_delay = 32

        # cellular network
        self.cell_length = 200. # m
        self.cell_number = 7

        # deep learning hyper-parameter

    def _cal_power_list_(self):
        power_gap = self.max_power * 2 / (self.power_level - 1)
        tmp_power = self.max_power
        for i in range(self.power_level):
            self.power_list.append(tmp_power)
            tmp_power -= power_gap

    def _cal_code_list_(self):
        # need to generate three sector of code matrix
        S = 16
        N = self.BS_antenna
        Q = self.codebook_size
        for n in range(N):
            for q in range(Q):
                self.beamform_list[q][n] = np.exp(1j*2*np.pi/S*(n*((q+Q/2)%Q)/(Q/S)))/np.sqrt(N)


if __name__ == "__main__":
    config = Config()
    print(config.power_list)
    print(config.beamform_list)
