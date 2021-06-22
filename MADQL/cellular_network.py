from data_generator_3sectors.config import Config
import math


class CellularNetwork:
    def __init__(self):
        self.c = Config()
        self._get_CU_position_()

    def _get_CU_position_(self):
        n = self.c.cellular_layers - 1
        self.num_of_CU = 1
        layers_BS = []
        for i in range(self.c.cellular_layers-1):
            self.num_of_CU += (i + 1) * 6
            layers_BS.append((i+1)*6)
        # print(layers_BS)
        self.CU_position = [[0, 0]]
        for i in range(self.c.cellular_layers - 1):
            theta = math.pi / 2.
            for j in range(layers_BS[i]):
                x = self.cell_length * 2 * (i + 1) * math.cos(theta)
                y = self.cell_length * 2 * (i + 1) * math.sin(theta)
                theta -= 2 * math.pi / layers_BS[i]
                self.CU_position.append([x, y])
        # print(self.CU_position)