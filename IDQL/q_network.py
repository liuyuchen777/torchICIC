import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from hyper_parameter import *


class BSNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BSNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, nodes_num[0], bias=True)
        self.hidden_layer1 = nn.Linear(nodes_num[0], nodes_num[1], bias=True)
        self.hidden_layer2 = nn.Linear(nodes_num[1], nodes_num[2], bias=True)
        self.hidden_layer3 = nn.Linear(nodes_num[2], nodes_num[3], bias=True)
        self.output_layer = nn.Linear(nodes_num[3], output_dim, bias=True)

    def forward(self, x):
        out = F.relu(self.input_layer(x))
        out = F.relu(self.hidden_layer1(out))
        out = F.relu(self.hidden_layer2(out))
        out = F.relu(self.hidden_layer3(out))
        return self.output_layer(out)


class QNetwork(object):
    def __init__(self, state_size, action_size, path=""):
        """
        init function of Q-Network
        :param state_size: input_dim of SDL
        :param action_size: output_dim of SDL
        :param path: model load path, if path="", train mode
        """
        # model define
        if path == "":
            self.model = BSNet(state_size, action_size)
        else:
            self.model = torch.load(path)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss = torch.nn.MSELoss()

    def train(self, x, y):
        """
        train function wrapper, each train step on a minibatch, x, y is tensor
        :param x: input state
        :param y: output action
        :return: MSE loss
        """
        # convert x, y to tensor
        x_tensor = torch.from_numpy(x.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.float32))

        for _ in range(num_of_epochs):
            # predict and compute loss
            y_predicted = self.model(x_tensor)
            loss = self.loss(y_tensor, y_predicted)
            # update network
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def predict(self, x):
        # convert x to tensor
        x_tensor = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            y_predict = self.model(x_tensor)

        return torch.detach(y_predict).numpy()

    def save(self, save_path):
        torch.save(self.model, save_path)
