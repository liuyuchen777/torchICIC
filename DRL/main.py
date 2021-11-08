from RL_simulator import RLSimulator
import numpy as np
from hyper_parameter import *
from const import *
from dqn_agent import Agent
from logger import Logger

if __name__ == '__main__':
    mylog = Logger()
    sim = RLSimulator(mylog)

    state_size = sim.env.get_state_size()
    action_size = sim.env.get_action_size()

    # start train
    max_rewards = np.empty(num_exp)
    for i in range(num_exp):
        mylog.log('start overall process %d:' % (i+1))
        all_agents = []
        for agent in range(bs_num):
            all_agents.append(Agent(state_size, action_size, agent))
        max_rewards[i] = sim.train(all_agents)

    mylog.log(f'Max Rewards is: {max_rewards}')
    mylog.log(f'Mean of Max Rewards is: {np.mean(max_rewards)}')
    mylog.log("-----------------------------done---------------------------------")