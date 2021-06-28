""" DTDE DRL-based scheme """
# main function of DDBC

from cellular_network import CellularNetwork as CN
import json
import random
import numpy as np
import scipy.io as sio
from config import Config
import os
os.environ['MKL_NUM_THREADS'] = '1'


c = Config()
random.seed(c.random_seed)
np.random.seed(c.random_seed)
cn = CN()
utility = []
cn.draw_topology()
rate_m = []

for i in range(c.total_slots):
    print("Time Slot: ", i)
    # observe environment state
    s = cn.observe()
    # choose action base observation and exchanged information
    # where is information exchange process?
    actions = cn.choose_actions(s)
    cn.update(ir_change=False, actions=actions)
    utility.append(cn.get_ave_utility())
    rate_m.append(cn.get_all_rates())
    # update channel state information
    cn.update(ir_change=True)
    r = cn.give_rewards()
    s_ = cn.observe()
    cn.save_transitions(s, actions, r, s_)

    if i > 256:
        # when time slot > 256, start train network
        cn.train_dqns()


# save data
filename = 'data/drl_performance.json'
with open(filename, 'w') as f:
    # utility saved in 'data/drl_performance.json'
    json.dump(utility, f)
rate_m = np.array(rate_m)
# data rate saved in 'rates/drl_rates.mat'
sio.savemat('rates/drl_rates.mat', {'drl_rates': rate_m})
