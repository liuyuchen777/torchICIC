import torch.nn as nn

from config import *


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # layers
        self.inputLayer = nn.Linear(INPUT_LAYER, HIDDEN_LAYERS[0])
        self.hiddenLayer1 = nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1])
        self.hiddenLayer2 = nn.Linear(HIDDEN_LAYERS[1], HIDDEN_LAYERS[2])
        self.outputLayer = nn.Linear(HIDDEN_LAYERS[2], OUTPUT_LAYER)

    def forward(self, input):
        # flow1
        out = self.inputLayer(input)
        out = self.hiddenLayer1(out)
        out = self.hiddenLayer2(out)
        out = self.outputLayer(out)
        return out
