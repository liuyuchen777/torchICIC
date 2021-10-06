import random
from Config import Config


class MemoryPool:
    def __init__(self, maxSize=Config().mpMaxSize):
        self.maxSize = maxSize
        self.pool = []

    def push(self, record):
        if self.getSize() >= self.maxSize:
            self.pool.pop(0)
            self.pool.append(record)
        else:
            self.pool.append(record)

    def getSize(self):
        return len(self.pool)

    def getBatch(self, size=Config().batchSize):
        random.shuffle(self.pool)
        return self.pool[:size]
