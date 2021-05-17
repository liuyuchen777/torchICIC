import numpy as np
import os
import time
from q_network import QNetwork
from memory_models import MemoryPool
from hyper_parameter import *
import torch


class Agent(object):
    def __init__(self, state_size, action_size, agent_id, path=""):
        """
        init function for independent DQN agent
        :param state_size:
        :param action_size:
        :param agent_id:
        :param path:
        """
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.memory_pool = MemoryPool(memory_pool_size)
        if path == "":
            self.q_network = QNetwork(self.state_size, self.action_size, path="")
        else:
            self.q_network = QNetwork(self.state_size, self.action_size, path=path)

    def select_action(self, state, mode=True):
        """
        select action based on epsilon-greedy policy
        :param state: state of radio_wave
        :param mode: true is for train, false is for test
        :return: action
        """
        state = state.reshape(1, -1)
        if not mode:
            # Test Mode: directly choose optimal action
            return np.argmax(self.q_network.predict(state))
        else:
            if np.random.rand() <= self.epsilon:
                return np.random.randint(self.action_size)
            else:
                # need to convert tensor to np
                return np.argmax(self.q_network.predict(state))

    def get_td_target(self, batch):
        """
        generate train data
        :param batch: (state, actions, reward)
        :return: network input x and network output y
        """
        states = np.array([b[0] for b in batch])
        p = self.q_network.predict(states)
        # generate input x and output y
        x = np.zeros((batch_size, self.state_size))
        y = np.zeros((batch_size, self.action_size))
        for i, b in enumerate(batch):
            state = b[0]
            action = b[1][self.agent_id]
            reward = b[2]

            temp = p[i]
            temp[action] = reward

            x[i] = state
            y[i] = temp

        return [x, y]

    def remember(self, sample):
        self.memory_pool.remember(sample)

    def decay_epsilon(self):
        """
        decay epsilon value
        :return: None
        """
        self.epsilon = max(0.01, self.epsilon * epsilon_decay)

    def replay(self):
        x, y = self.get_td_target(self.memory_pool.get())
        # x, y need to be tensor
        self.q_network.train(x, y)

    def save_model(self, model_id):
        dir_path = model_path + str(model_id)
        try:
            os.mkdir(dir_path)
        except:
            pass
        path = dir_path + "/" + str(self.agent_id) + ".pth"
        self.q_network.save(path)
