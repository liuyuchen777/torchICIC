""""
Author: Liu Yuchen
Date: 2021-04-12 13:57:44
LastEditors: Liu Yuchen
LastEditTime: 2021-04-12 13:58:27
Description: torch version of joint transmit power and 3D beamforming optimization
FilePath: /Local_Lab/DRL_for_ICIC/Torch_ICIC/model_3d_beam_power.py
GitHub: https://github.com/liuyuchen777
"""

# framework package
import time

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

# self define package
from const import *
from utils import *
from hyper_parameter import *

print(torch.__version__)


# define network structure
class BSNet(nn.Module):
    def __init__(self, input_dim, output_dim, drop_rate=0.5):
        super(BSNet, self).__init__()
        # input layer
        self.layer1 = nn.Linear(input_dim, full_con_layer1_unit, bias=True)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.constant_(self.layer1.bias, 0)
        # hidden layer
        self.drop2 = nn.Dropout(p=drop_rate)
        self.layer2 = nn.Linear(full_con_layer1_unit, full_con_layer2_unit, bias=True)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.constant_(self.layer2.bias, 0)
        self.drop3 = nn.Dropout(p=drop_rate)
        self.layer3 = nn.Linear(full_con_layer1_unit, full_con_layer2_unit, bias=True)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        torch.nn.init.constant_(self.layer3.bias, 0)
        self.drop4 = nn.Dropout(p=drop_rate)
        self.layer4 = nn.Linear(full_con_layer1_unit, full_con_layer2_unit, bias=True)
        torch.nn.init.xavier_uniform_(self.layer4.weight)
        torch.nn.init.constant_(self.layer4.bias, 0)
        self.drop5 = nn.Dropout(p=drop_rate)
        self.layer5 = nn.Linear(full_con_layer1_unit, full_con_layer2_unit, bias=True)
        torch.nn.init.xavier_uniform_(self.layer5.weight)
        torch.nn.init.constant_(self.layer5.bias, 0)
        # output layer
        self.layer6 = nn.Linear(full_con_layer2_unit, output_dim, bias=True)
        torch.nn.init.xavier_uniform_(self.layer6.weight)
        torch.nn.init.constant_(self.layer6.bias, 0)

    def forward(self, x):
        # input layer
        out = F.relu(self.layer1(x))
        # hidden layer
        out = self.drop2(out)
        out = F.relu(self.layer2(out))
        out = self.drop3(out)
        out = F.relu(self.layer3(out))
        out = self.drop4(out)
        out = F.relu(self.layer4(out))
        out = self.drop5(out)
        out = F.relu(self.layer5(out))
        # output layer
        out = self.layer6(out)

        return out


def beam_loss(y_predict, y):
    loss = nn.CrossEntropyLoss()
    total_beam_loss = 0
    for bs in range(bs_num):
        true_beam = y[:, bs_num * len(P_cb) + bs * len(precoding_matrices):bs_num * len(P_cb) + (bs + 1) * len(
            precoding_matrices)]
        true_beam_label = true_beam.argmax(dim=1)
        each_beam = y_predict[:, bs_num * len(P_cb) + bs * len(precoding_matrices):bs_num * len(P_cb) + (bs + 1) * len(
            precoding_matrices)]
        total_beam_loss += loss(each_beam, true_beam_label)

    return total_beam_loss


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    else:
        print(m, "not define!")


def power_loss(y_predict, y):
    loss = nn.CrossEntropyLoss()
    total_power_loss = 0
    for bs in range(bs_num):
        true_power = y[:, bs * len(P_cb):(bs + 1) * len(P_cb)]
        true_power_label = true_power.argmax(dim=1)
        each_power = y_predict[:, bs * len(P_cb):(bs + 1) * len(P_cb)]
        # 根据each_power_softmax和true power计算cross entropy
        # nn.CrossEntropyLoss doesn't need to do softmax
        total_power_loss += loss(each_power, true_power_label)

    # 计算total_power_loss
    return total_power_loss


def total_loss(y_predicted, y):
    b_loss = beam_loss(y_predicted, y)
    p_loss = power_loss(y_predicted, y)
    t_loss = theta * b_loss + (1 - theta) * p_loss
    return b_loss, p_loss, t_loss


# main function
if __name__ == "__main__":
    mylog = Logger()
    # 0) load data
    x_train, x_test, y_train, y_test = data_loader(data_path)
    # print(y_train.shape)    # (16000, 39)
    x_samples, x_features = x_train.shape
    y_samples, y_features = y_train.shape
    # convert to torch tensor
    x_train = torch.from_numpy(x_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    x_test = torch.from_numpy(x_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    # print(x_train)
    # print('requires_grad:', x_train.requires_grad)
    # print('test')

    # 1) define optimizer and loss function
    if mode == "initial":
        model = BSNet(x_features, y_features)
        # model.apply(weight_init)
    else:
        model = torch.load(retrain_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_beta)

    # 2) training loop
    # use mini batch, train data size is 16000, too large in 1 epoch
    beam_loss_record = []
    power_loss_record = []
    validate_beam_loss_record = []
    validate_power_loss_record = []

    for epoch in range(num_epochs):
        epoch_loss = 0.
        epoch_beam_loss = 0.
        epoch_power_loss = 0.
        num_minibatches = int(x_train.shape[0] / mini_batch_size)
        mini_batches = random_mini_batches(x_train, y_train, mini_batch_size)

        for (minibatch_x, minibatch_y) in mini_batches:
            # forward pass
            y_predicted = model(minibatch_x)

            # backward pass, cal gradient
            batch_beam_loss, batch_power_loss, batch_loss = total_loss(y_predicted, minibatch_y)
            with torch.no_grad():
                epoch_beam_loss += batch_beam_loss / num_minibatches
                epoch_power_loss += batch_power_loss / num_minibatches
                epoch_loss += batch_loss / num_minibatches

            # update
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # print out loss
        mylog.log(f'epoch = {epoch+1}: ')
        mylog.log(f'    epoch_loss ----- {epoch_loss}')
        mylog.log(f'    epoch_power_loss ----- {epoch_power_loss}')
        mylog.log(f'    epoch_beam_loss ----- {epoch_beam_loss}')
        # cross validation
        with torch.no_grad():
            y_validate = model(x_test)
            validate_beam_loss = beam_loss(y_validate, y_test)
            validate_power_loss = power_loss(y_validate, y_test)
            validate_loss = theta * validate_beam_loss + (1 - theta) * validate_power_loss
        # print out validation loss
        mylog.log(f'  cross validation: ')
        mylog.log(f'    validation_loss ----- {validate_loss}')
        mylog.log(f'    validation_power_loss ----- {validate_power_loss}')
        mylog.log(f'    validation_beam_loss ----- {validate_beam_loss}')
        # record loss
        beam_loss_record.append(epoch_beam_loss)
        power_loss_record.append(epoch_power_loss)
        validate_beam_loss_record.append(validate_beam_loss)
        validate_power_loss_record.append(validate_power_loss)

    # 3) plt

    # 4) save model
    save_path_now = save_path + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + ".pth"
    torch.save(model, save_path_now)
    logging.info("----------------------------FINISH----------------------------------\n\n")