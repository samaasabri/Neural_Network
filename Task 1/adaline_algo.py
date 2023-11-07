import numpy as np


def Adaline(X, y, learning_rate=0.01, epochs=1000,mse_threshold=0.01,b = 0,use_bias=False):
    samples, feat = X.shape
    W = np.random.rand(feat)

    for _ in range(epochs):
        errs=[]
        for i, x in enumerate(X):
            y_pred = np.dot(x, W) + b
            err=(y[i] - y_pred)
            errs.append(err)
            upt = learning_rate * err
            W += upt * x
            if use_bias:
                b+=upt
        mse = np.mean(errs)**2
        if(mse<mse_threshold):
            break
        

    return W,b