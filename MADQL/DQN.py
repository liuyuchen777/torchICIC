import torch
import torch.nn.functional as F
import torch.nn as nn

from Config import Config


class DQN(nn.Module):
    """
    input:
        two-flow
        flow1 -> 12 * 24
        flow2 -> 63 * 8
    output:
        [2 (keep current power || change to another power) * 2 (keep current beamformer || change to another beamformer)] ^ 3
        64 * 1
    """
    def __init__(self):
        super(DQN, self).__init__()
        self.config = Config()
        # layers
        # flow1
        self.F1Conv1 = nn.Conv2d(1, 3, 3)
        self.F1Pool1 = nn.MaxPool2d(2, 2)
        self.F1Conv2 = nn.Conv2d(3, 5, 3)
        self.F1FC1 = nn.Linear(5 * 3 * 9, 80)
        # flow2
        self.F2Conv1 = nn.Conv2d(1, 3, 3)
        self.F2Pool1 = nn.MaxPool2d(2, 2)
        self.F2Conv2 = nn.Conv2d(3, 5, 3)
        self.F2FC1 = nn.Linear(5 * 28 * 1, 120)
        # compose flow1 and flow2
        self.ComFC1 = nn.Linear(200, 120)
        self.ComFC2 = nn.Linear(120, 64)

    def forward(self, flow1, flow2):
        # flow1
        flow1 = F.relu(self.F1Conv1(flow1))
        flow1 = self.F1Pool1(flow1)
        flow1 = F.relu(self.F1Conv2(flow1))
        flow1 = torch.flatten(flow1, 1)
        flow1 = F.relu(self.F1FC1(flow1))
        # flow2
        flow2 = F.relu(self.F2Conv1(flow2))
        flow2 = self.F2Pool1(flow2)
        flow2 = F.relu(self.F2Conv2(flow2))
        flow2 = torch.flatten(flow2, 1)
        flow2 = F.relu(self.F2FC1(flow2))
        # compose
        out = torch.cat((flow1, flow2), 1)
        out = self.ComFC1(out)
        out = self.ComFC2(out)
        return out
