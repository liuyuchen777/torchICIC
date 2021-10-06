from Config import Config


class Sector:
    def __init__(self, index, CUIndex, pos):
        self.index = index
        self.CUIndex = CUIndex
        self.pos = pos  # [x, y, z]
        self.config = Config()
