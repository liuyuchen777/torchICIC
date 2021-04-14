import numpy as np

# f = w * x

# f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

W = 0.0


# model prediction
def forward(x):
    return W * x


# loss = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()


# gradient
# MSE = 1/N * (w * x - y)**2
# dj/dw = 1/N * 2x * (w * x - y)
def gradient(x, y, y_pred):
    return np.dot(2 * x, y_pred - y).mean()


if __name__ == "__main__":
    print(f'Prediction before training: f(5) = {forward(5):.3f}')

    # training
    learning_rate = 0.01
    n_iter = 100

    for epoch in range(n_iter):
        # predication = forward pass
        y_predict = forward(X)

        # loss
        l = loss(Y, y_predict)

        # gradient
        dw = gradient(X, Y, y_predict)

        # update weight
        W -= dw * learning_rate

        # print training process
        if epoch % 10 == 0:
            print(f'epoch {epoch+1}: w = {W:.3f}, loss = {l:.3f}')

    print(f'Prediction before training: f(5) = {forward(5):.3f}')
