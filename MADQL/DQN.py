import torch.nn as nn

from Config import Config


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.config = Config()
        # layers
        self.inputLayer = nn.Linear(self.config.inputLayer, self.config.hiddenLayers[0])
        self.hiddenLayer1 = nn.Linear(self.config.hiddenLayers[0], self.config.hiddenLayers[1])
        self.hiddenLayer2 = nn.Linear(self.config.hiddenLayers[1], self.config.hiddenLayers[2])
        self.outputLayer = nn.Linear(self.config.hiddenLayers[2], self.config.outputLayer)

    def forward(self, input):
        # flow1
        out = self.inputLayer(input)
        out = self.hiddenLayer1(out)
        out = self.hiddenLayer2(out)
        out = self.outputLayer(out)
        return out
