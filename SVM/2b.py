import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)
    
    x_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    x_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
    
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)
    
    return x_train, y_train, x_test, y_test

def vector_norm(v):
    return sum(abs(v_i) for v_i in v)

def svm_primal_sgd(x, y, C, gamma_0, a, T):
    w = np.zeros(x.shape[1])
    n_samples = len(x)
    b = 0

    for epoch in range(T):
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        x, y = x[indices], y[indices]

        for i in range(n_samples):
            t = epoch * n_samples + i + 1
            gamma_t = gamma_0 / (1 + t)

            if y[i] * (np.dot(w, x[i]) + b) <= 1:
                w = w - gamma_t * w + gamma_t * C * y[i] * x[i]
                b = b + gamma_t * C * y[i]
            else:
                w = (1 - gamma_t) * w
    
    return w, b

train_path = "bank-note/train.csv"
test_path = "bank-note/test.csv"

x_train, y_train, x_test, y_test = load_data(train_path, test_path)

C_values = [100/873, 500/873, 700/873]
gamma_0 = 5
a = 10
T = 100

for C in C_values:
    print(f"Training with C = {C}")
    w, b = svm_primal_sgd(x_train, y_train, C, gamma_0, a, T)

    train_error = np.mean(np.sign(np.dot(x_train, w) + b) != y_train)
    test_error = np.mean(np.sign(np.dot(x_test, w) + b) != y_test)
    print(f"Weights: {w}")
    print(f"Bias: {b}")
    print(f"Training Error: {train_error:.4f}")
    print(f"Test Error: {test_error:.4f}\n")

