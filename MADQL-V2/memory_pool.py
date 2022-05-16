import random
from collections import deque

from config import *


class MemoryPool:
    def __init__(self, maxSize=MP_MAX_SIZE):
        self.maxSize = maxSize
        self.pool = deque([], maxlen=MP_MAX_SIZE)

    def push(self, record):
        self.pool.append(record)

    def getSize(self):
        return len(self.pool)

    def getBatch(self, size=BATCH_SIZE//(CELL_NUMBER*3)):
        return random.sample(self.pool, size)
