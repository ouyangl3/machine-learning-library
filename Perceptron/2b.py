import numpy as np
import pandas as pd

def voted_perceptron(x, y, epochs=10, r=0.1):
    weights = np.zeros(x.shape[1])
    bias = 0
    weight_vectors = []
    count = 0

    for epoch in range(epochs):
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x, y = x[indices], y[indices]
        for i in range(len(x)):
            if y[i] * (np.dot(x[i], weights) + bias) <= 0:
                weight_vectors.append((weights.copy(), bias, count))
                weights += r * y[i] * x[i]
                bias += r * y[i]
                count = 1
            else:
                count += 1

    weight_vectors.append((weights.copy(), bias, count))
    return weight_vectors

def voted_predict(x, weight_vectors):
    predictions = np.zeros(len(x))
    for w, b, c in weight_vectors:
        predictions += c * np.sign(np.dot(x, w) + b)
    return np.sign(predictions)

train_data = pd.read_csv("bank-note/train.csv", header=None)
test_data = pd.read_csv("bank-note/test.csv", header=None)

x_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
x_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

y_train = np.where(y_train == 0, -1, 1)
y_test = np.where(y_test == 0, -1, 1)

weight_vectors = voted_perceptron(x_train, y_train, epochs=10)

for w, b, c in weight_vectors:
    print("Weights:", w, "Bias:", b, "Count:", c)

y_pred = voted_predict(x_test, weight_vectors)

error_rate = np.mean(y_pred != y_test)
print("Average prediction error:", error_rate)
