'''
Author: Liu Yuchen
Date: 2021-06-22 15:29:47
LastEditors: Liu Yuchen
LastEditTime: 2021-06-22 15:57:23
Description: config file for communication and hyperparameters in DRL
FilePath: /torch_ICIC/MADQL/config.py
GitHub: https://github.com/liuyuchen777
'''

class Config:
    def __init__(self):
        # base station and user terminal
        self.BS_antenna = 16
        self.UT_antenna = 4
        self.BS_height = 10.
        self.UT_height = 1.5
        self.CU_length = 30.

        # power level list
        self.max_power = 10.     # num
        self.power_level = 5
        self.power_list = []
        self._get_power_list_()
        # beamforming vector list
        self.codebook_size = 10
        self.beamform_list = []
        self._get_code_list_()

        # channel
        self.Rician_factor = 10
        self.LOS_path = 6
        self.all_component = 8
        self.max_delay = 32

        # cellular network
        self.cell_length = 30.
        self.cellular_layers = 2

    def _get_power_list_(self):
        power_gap = self.max_power * 2 / (self.power_level - 1)
        tmp_power = self.max_power
        for i in range(self.power_level):
            self.power_list.append(tmp_power)
            tmp_power -= power_gap
        # print(self.power_list)

    def _get_code_list_(self):
        # need to generate three sector of code matrix
        print("under construct")


if __name__ == "__main__":
    c = Config()
