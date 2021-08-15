import numpy as np


class MemoryPool:
    def __init__(self, pool_size=10000):
        self.pool_size = pool_size
        self.pool = []

    def push(self, record):
        self.pool.append(record)

    def get_size(self):
        return len(self.pool)

