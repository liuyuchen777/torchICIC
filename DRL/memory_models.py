import random


# Q-learning中记忆之前所有case的经验池
class MemoryPool(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def reset(self):
        self.memory = []

    def remember(self, sample):
        self.memory.append(sample)

    def sample(self, n):
        n = min(n, len(self.memory))
        sample_batch = random.sample(self.memory, n)
        return sample_batch

    def get(self):
        return self.memory
