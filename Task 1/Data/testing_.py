# -*- coding: utf-8 -*-
"""Testing .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Sw7TND8L9ptDne2h_0Hgn3VVBx5qZFgT
"""

import numpy as np
import pandas as pd


# Function take one sample x
# When calling loop on number of test samples

# X shape >>(2,1)
# W sahpe>>(1,2)
def test_fn(x_test, w, b):
    y_new = np.dot(x_test, w) + b
    return np.where(y_new >= 0, 1, -1)


######## Testing the function ########

# W=np.random.rand(1,2)
# print(W)

# X=np.array([[1,1]])
# X=X.reshape(2,1)
# print(X)

# y=testing(X,W)
# print(y)

# C1 and C2 class names entered from GUI
def evaluation(actual_res, Target, C1, C2):
    # TP , TN , FP , FN
    matrix = np.zeros((2, 2))
    for i in range(len(actual_res)):
        if actual_res[i] == 1 and Target[i] == 1:
            matrix[0, 0] += 1  # True Positives
        elif actual_res[i] == 1 and Target[i] == -1:
            matrix[0, 1] += 1  # False Positives
        elif actual_res[i] == -1 and Target[i] == 1:
            matrix[1, 0] += 1  # False Negatives
        elif actual_res[i] == -1 and Target[i] == -1:
            matrix[1, 1] += 1  # True Negatives

    accuracy = (matrix[0, 0] + matrix[1, 1]) / (len(actual_res))

    df = pd.DataFrame(matrix, columns=[C1, C2], index=[C1, C2])

    print("Accuracy:", accuracy)
    print("confusion_matrix:\n", df)

    return

######## Testing the function ########

# y=[1,-1,-1]
# T=[1,-1,-1]
# evaluation(y , T , "Sira", "Bomay")
