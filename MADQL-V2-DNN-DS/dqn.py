import torch.nn as nn
import torch.nn.functional as F

from config import *


class DQN(nn.Module):
    def __init__(self, inputLayer, outputLayer):
        super(DQN, self).__init__()
        self.input_layer = nn.Linear(inputLayer, HIDDEN_LAYER[0], bias=True)
        self.hidden_layer1 = nn.Linear(HIDDEN_LAYER[0], HIDDEN_LAYER[1], bias=True)
        self.hidden_layer2 = nn.Linear(HIDDEN_LAYER[1], HIDDEN_LAYER[2], bias=True)
        self.hidden_layer3 = nn.Linear(HIDDEN_LAYER[2], HIDDEN_LAYER[3], bias=True)
        self.output_layer = nn.Linear(HIDDEN_LAYER[3], outputLayer, bias=True)

    def forward(self, x):
        out = F.relu(self.input_layer(x))
        out = F.relu(self.hidden_layer1(out))
        out = F.relu(self.hidden_layer2(out))
        out = F.relu(self.hidden_layer3(out))
        return self.output_layer(out)