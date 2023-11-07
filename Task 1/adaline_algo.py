import numpy as np


def Adaline(X, y, learning_rate=0.01, epochs=1000):
   samples, feat = X.shape
   W = np.random.rand(feat)
   b = 0

   for _ in range(epochs):
       for i, x in enumerate(X):
           y_pred = np.dot(x, W) + b
           #y_pred = 1 if y_pred >= 0 else -1
           upt = learning_rate * (y[i] - y_pred)
           W += upt * x
           #b += upt

   return W