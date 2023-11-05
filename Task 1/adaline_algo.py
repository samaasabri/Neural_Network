import numpy as np

def Adaline(X, y, lr, epochs,b = 0):
    samples, features = X.shape
    w = np.zeros(features)

    def predict(X):
        output = np.dot(X,w)+b
        return np.where(output>=0,1,0)
    
    for _ in range(epochs):
        ynew = predict(X)
        err=y - ynew
        w +=  lr * X.T.dot(err)
        # b += lr*np.sum(err)
    return w