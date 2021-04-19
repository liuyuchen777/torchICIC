import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0） prepare data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0], 1)   # resize Y

n_samples, n_features = X.shape

# 1) define model
input_size = n_features
output_size = 1
# model = nn.Linear(input_size, output_size)


# change to multi-layer model
class ThreeLayerNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ThreeLayerNN, self).__init__()
        # self.lin = nn.Linear(input_dim, output_dim)
        input_layer_unit = 10
        hidden_layer_unit = 20
        self.inLayer = nn.Linear(input_dim, input_layer_unit)
        self.ReLu = nn.ReLU()
        self.hideLayer = nn.Linear(input_layer_unit, hidden_layer_unit)
        self.outLayer = nn.Linear(hidden_layer_unit, output_dim)

    def forward(self, x):
        tmp1 = self.inLayer(x)
        tmp1 = self.ReLu(tmp1)
        tmp1 = self.hideLayer(tmp1)
        tmp1 = self.ReLu(tmp1)
        return self.outLayer(tmp1)


model = ThreeLayerNN(input_size, output_size)

# 2) define loss function and optimizer
learning_rate = 0.001
"""
when learning_rate=0.01, loss will be nan; 0.001-0.0001 is fine 
but smaller than 0.0001 will make a horizontal line
"""
criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epoch = 100
for epoch in range(num_epoch):
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, Y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()
    model.zero_grad()

    if epoch % 1 == 0:
        print(f'epoch: {epoch+1}, loss={loss.item():.4f}')

# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()
