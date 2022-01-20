import random

from Config import Config
import numpy as np

class MemoryPool:
    def __init__(self, maxSize=Config().mpMaxSize):
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

    def getBatch(self, size=Config().batchSize):
        random.shuffle(self.pool)
        return self.pool[:size]
