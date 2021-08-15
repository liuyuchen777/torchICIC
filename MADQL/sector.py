
from config import Config


class Sector:
    def __init__(self, index, CU_index, pos, logger, config=Config()):
        self.index = index
        self.CU_index = CU_index
        self.pos = pos  # [x, y, z]
        self.logger = logger
        self.config = config
