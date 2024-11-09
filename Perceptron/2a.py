import numpy as np
import pandas as pd

def perceptron(x, y, epochs=10, r=0.1):
    weights = np.zeros(x.shape[1])
    bias = 0

    for epoch in range(epochs):
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x, y = x[indices], y[indices]
        for i in range(len(x)):
            if y[i] * (np.dot(x[i], weights) + bias) <= 0:
                weights += r * y[i] * x[i]
                bias += r * y[i]

    return weights, bias

def predict(x, weights, bias):
    return np.sign(np.dot(x, weights) + bias)

train_data = pd.read_csv("bank-note/train.csv", header=None)
test_data = pd.read_csv("bank-note/test.csv", header=None)

x_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
x_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

weights, bias = perceptron(x_train, y_train, epochs=10)
print("Learned weights:", weights)
print("Learned bias:", bias)

y_pred = predict(x_test, weights, bias)

error_rate = np.mean(y_pred != y_test)
print("Average prediction error:", error_rate)
