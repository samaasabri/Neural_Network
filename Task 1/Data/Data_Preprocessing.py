import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_excel(r"D:\FCIS\Neural-Network\Project\Neural_Network\Task 1\Data\Dry_Bean_Dataset.xlsx")

data['MinorAxisLength'].fillna(value=data['MinorAxisLength'].mean(), inplace=True)



def plot_data(X_test, y_test, w, b, lable):
    plt.title(lable)
    # Plotting the data points and decision boundary if possible
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
    # Check if the decision boundary is linear
    if w[1] == 0:
        plt.axvline(-b / w[0])
    else:
        x = np.linspace(np.min(X_test[:, 0]), np.max(X_test[:, 0]), 100)
        y = -(w[0] * x + b) / w[1]
        plt.plot(x, y)
    plt.show()


def normalization(data):
    data = (data - data.min()) / (data.max() - data.min()) * (20 - 7) - 1
    return data


def Split_my_data(choice,feature1,feature2):
    # C1 & C2
    if choice == 1:
        x = data.iloc[:100, :-1]
        x = x[[feature1, feature2]].values
        x = normalization(x)
        y = data.iloc[:100, -1]
        y = [1 if val == "BOMBAY" else -1 for val in y]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=10)
    # C1 & C3
    elif choice == 2:
        d1 = data.iloc[:50, :]
        d2 = data.iloc[100:150, :]
        d = pd.concat([d1, d2])
        x = d.iloc[:, :-1]
        x = x[[feature1, feature2]].values
        x = normalization(x)
        y = d.iloc[:, -1]
        y = [1 if val == "BOMBAY" else -1 for val in y]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=10)
    # C2 & C3
    elif choice == 3:
        x = data.iloc[50:, :-1]
        x = x[[feature1, feature2]].values
        x = normalization(x)
        y = data.iloc[50:, -1]
        y = [1 if val == "CALI" else -1 for val in y]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=10)

    return x_train, x_test, y_train, y_test
