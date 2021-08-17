import random
from config import Config


class MemoryPool:
    def __init__(self, max_size=Config().mp_max_size):
        self.max_size = max_size
        self.pool = []

    def push(self, record):
        if self.get_size() >= self.max_size:
            self.pool.pop(0)
            self.pool.append(record)
        else:
            self.pool.append(record)

    def get_size(self):
        return len(self.pool)

    def get_batch(self, size=Config().batch_size):
        random.shuffle(self.pool)
        return self.pool[:size]
