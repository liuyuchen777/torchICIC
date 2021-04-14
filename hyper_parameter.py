# define hyper parameter of system
reg_beta = 0.01   # 正则化，加在损失函数里
dropout_rate = 0.5
num_epochs = 100    # num of epochs
learning_rate = 0.0001
mini_batch_size = 16
theta = 0.5     # loss function balance between beam loss and power loss
full_con_layer1_unit = 1024
full_con_layer2_unit = 1024
data_path = "../data/train_data_3d/greedy_capacity(1752)b_p_1stream_max_desire.npz"
