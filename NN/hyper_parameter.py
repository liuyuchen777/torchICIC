'''
Author: Liu Yuchen
Date: 2021-06-22 15:37:44
LastEditors: Liu Yuchen
LastEditTime: 2021-06-22 15:39:23
Description: 
FilePath: /torch_ICIC/NN/hyper_parameter.py
GitHub: https://github.com/liuyuchen777
'''
# define hyper parameter of system
reg_beta = 1e-5   # 正则化，加在损失函数里
dropout_rate = 0.5
num_epochs = 100    # num of epochs
learning_rate = 0.0001
mini_batch_size = 16
theta = 0.5     # loss function balance between beam loss and power loss
full_con_layer1_unit = 1024
full_con_layer2_unit = 1024
data_path = "./data/train_data_3d/greedy_capacity_b_p_ES_train.npz"
save_path = "./checkpoints/"
log_path = "./log/"
mode = "initial"
retrain_path = "./checkpoints/2021-04-22_18-59-43.pth"
