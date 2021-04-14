import torch
import torch.nn as nn
import numpy as np


def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

"""
# Y mush be one hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
Y = np.array([1, 0, 0])

# Y_predicted has probability
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])

# numpy version
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'loss1 numpy: {l1:.3f}')
print(f'loss2 numpy: {l2:.3f}')
"""

# torch version
Y = torch.tensor([0])
loss = nn.CrossEntropyLoss()
# n_samples * n_classes = 1 * 3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l21 = loss(Y_pred_good, Y)
l22 = loss(Y_pred_bad, Y)

print(l21.item())
print(l22.item())