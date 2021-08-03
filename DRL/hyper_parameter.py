'''
Author: Liu Yuchen
Date: 2021-06-14 20:00:47
LastEditors: Liu Yuchen
LastEditTime: 2021-06-22 15:36:21
Description: 
FilePath: /torch_ICIC/DRL/hyper_parameter.py
GitHub: https://github.com/liuyuchen777
'''

iteration = 100
batch_size = 1600
epsilon = 1
epsilon_decay = 0.96
minibatch_size = 16
num_of_epochs = 30
nodes_num = [1024, 1024, 1024, 1024]
epochs_num = 30
learning_rate = 0.001
memory_pool_size = batch_size
model_path = "../../model/RL_model/"
log_path = "./log/"
num_exp = 10
