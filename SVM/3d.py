import numpy as np
import pandas as pd

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)
    
    x_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
    x_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
    
    y_train = np.where(y_train == 0, -1, 1)
    y_test = np.where(y_test == 0, -1, 1)
    
    return x_train, y_train, x_test, y_test

def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2)**2 / gamma)

def kernel_perceptron_train(X, y, max_iterations, gamma):
    n_samples = X.shape[0]
    c = np.zeros(n_samples)

    for _ in range(max_iterations):
        for i in range(n_samples):
            prediction = 0
            for j in range(n_samples):
                prediction += c[j] * y[j] * gaussian_kernel(X[j], X[i], gamma)
            
            if y[i] * prediction <= 0:
                c[i] += 1
    
    return c

def kernel_perceptron_predict(X_train, y_train, X_test, c, gamma):
    predictions = []
    n_samples_train = X_train.shape[0]

    for x_new in X_test:
        score = 0
        for i in range(n_samples_train):
            score += c[i] * y_train[i] * gaussian_kernel(X_train[i], x_new, gamma)
        predictions.append(np.sign(score))
    
    return np.array(predictions)

train_path = "bank-note/train.csv"
test_path = "bank-note/test.csv"
x_train, y_train, x_test, y_test = load_data(train_path, test_path)

gamma_values = [0.1, 0.5, 1, 5, 100]
max_iterations = 100

for gamma in gamma_values:
    print(f"Training with gamma = {gamma}")
    
    c = kernel_perceptron_train(x_train, y_train, max_iterations, gamma)
    
    train_predictions = kernel_perceptron_predict(x_train, y_train, x_train, c, gamma)
    test_predictions = kernel_perceptron_predict(x_train, y_train, x_test, c, gamma)
    
    train_error = np.mean(train_predictions != y_train)
    test_error = np.mean(test_predictions != y_test)
    
    print(f"Training Error: {train_error:.4f}")
    print(f"Test Error: {test_error:.4f}")
    print("\n")
