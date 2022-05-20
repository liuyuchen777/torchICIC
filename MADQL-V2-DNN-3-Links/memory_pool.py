import random
from collections import deque

from config import *


class MemoryPool:
    def __init__(self, capacity=MP_MAX_SIZE):
        self.pool = deque([], maxlen=capacity)

    def push(self, record):
        self.pool.extend(record)

    def getSize(self):
        return len(self.pool)

    def getBatch(self, size=BATCH_SIZE):
        return random.sample(self.pool, size)
