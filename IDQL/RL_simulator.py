import numpy as np
import time
from utils import *
from env import Env
from dqn_agent import Agent
from const import bs_num
from hyper_parameter import *


class RLSimulator(object):
    def __init__(self, mylog):
        self.env = Env(bs_num)
        self.model_id = time.strftime("%Y-%m-%d")
        self.mylog = mylog

    def train(self, agents):
        max_reward = 0
        start = time.time()
        for i in range(iteration):
            # generate training data
            for _ in range(batch_size):
                # generate 1600 data items
                state, reward = self.env.reset()
                actions = []
                for agent in agents:
                    actions.append(agent.select_action(state))
                reward = self.env.step(actions)
                for agent in agents:
                    agent.remember((state, actions, reward))
            # update network
            for agent in agents:
                agent.replay()
                agent.memory_pool.reset()
                # decay epsilon (optional)
                # agent.decay_epsilon()
                agent.EPSILON = max(1 - ((i + 1) / 30.), 0.1)
            # cross validation
            rewards = self.test(agents)
            time_elapsed = time.time() - start
            self.mylog.log('iter= %d, reward= %.2f, time elapsed= %d (s)'
                           % (i+1, np.asscalar(np.mean(rewards)), time_elapsed))
            max_reward = max(max_reward, np.mean(rewards))

        self.mylog.log('max_reward= %.2f' % max_reward)
        # save model
        self.save_agents(agents)
        return max_reward

    def test(self, agents, num_of_data=1000, data=None):
        """
        test model, two mode, give data and don't have data
        :param agents: agents of network
        :param num_of_data: test data number
        :param data: test data
        :return: rewards list (dim=num_of_data)
        """
        if not data:
            rewards = np.empty(num_of_data)
            for i in range(num_of_data):
                state, _ = self.env.reset()
                actions = []
                for agent in agents:
                    actions.append(agent.select_action(state, False))
                rewards[i] = self.env.step(actions)
            return rewards
        else:
            states = data['X']
            G_all = np.expand_dims(data['G_all'], 1)
            num_data = states.shape[0]
            rewards = np.empty(num_data)
            for i in range(num_data):
                state, _ = self.env.set_state(states[i], G_all[i])
                actions = []
                for agent in agents:
                    actions.append(agent.select_action(state))
                rewards[i] = self.env.step(actions)
            return rewards

    def save_agents(self, agents):
        for agent in agents:
            agent.save_model(self.model_id)
