import random

from config import *


class MemoryPool:
    def __init__(self, maxSize=MP_MAX_SIZE):
        self.maxSize = maxSize
        self.pool = []
        self.size = 0

    def push(self, record):
        if self.getSize() >= self.maxSize:
            self.pool.pop(0)
            self.pool.append(record)
        else:
            self.pool.append(record)
            self.size += 1

    def getSize(self):
        return self.size

    def getBatch(self, size=BATCH_SIZE):
        return random.sample(self.pool, size)
