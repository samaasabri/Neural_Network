import numpy as np
# if bais check box get random value and pass it to the function


def perceptron(X, y, learning_rate=0.01, epochs=1000, b=0,use_bias=False):
    samples, feat = X.shape
    W = np.random.rand(feat)

    for _ in range(epochs):
        for i, x in enumerate(X):
            y_pred = np.dot(x, W) + b

            if y_pred >= 0:
                y_pred = 1
            else:
                y_pred = -1
            upt = learning_rate * (y[i] - y_pred)
            W += upt * x
            if use_bias:
                b+=upt

    return W,b