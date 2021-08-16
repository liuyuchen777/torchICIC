from config import Config
from utils import Logger


class Sector:
    def __init__(self, index, CU_index, pos):
        self.index = index
        self.CU_index = CU_index
        self.pos = pos  # [x, y, z]
        self.config = Config()
