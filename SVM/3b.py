import numpy as np
import pandas as pd
from scipy.optimize import minimize

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

def compute_kernel_matrix(X, gamma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)
    return K

def svm_dual_gaussian(X, y, C, gamma):
    n_samples = X.shape[0]
    K = compute_kernel_matrix(X, gamma)
    
    def objective(alpha):
        return 0.5 * np.dot(alpha * y, np.dot(K, alpha * y)) - np.sum(alpha)
    
    constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, y)}
    
    bounds = [(0, C) for _ in range(n_samples)]
    
    initial_alpha = np.zeros(n_samples)
    
    result = minimize(
        objective,
        initial_alpha,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 100}
    )
    
    alpha = result.x
    
    support_vector_indices = np.where(alpha > 1e-5)[0]
    b = np.mean([y[i] - np.sum(alpha[support_vector_indices] * y[support_vector_indices] *
                                K[support_vector_indices, i]) for i in support_vector_indices])
    
    return alpha, b, support_vector_indices

def predict(X_train, y_train, X_test, alpha, b, gamma):
    n_samples = X_train.shape[0]
    predictions = []
    for x in X_test:
        decision = np.sum([
            alpha[i] * y_train[i] * gaussian_kernel(X_train[i], x, gamma)
            for i in range(n_samples)
        ]) + b
        predictions.append(np.sign(decision))
    return np.array(predictions)

train_path = "bank-note/train.csv"
test_path = "bank-note/test.csv"
x_train, y_train, x_test, y_test = load_data(train_path, test_path)

C_values = [100/873, 500/873, 700/873]
gamma_values = [0.1, 0.5, 1, 5, 100]

for gamma in gamma_values:
    for C in C_values:
        print(f"Training with gamma = {gamma}, C = {C:.4f}")
        
        alpha, b, sv_indices = svm_dual_gaussian(x_train, y_train, C, gamma)
        
        train_predictions = predict(x_train, y_train, x_train, alpha, b, gamma)
        test_predictions = predict(x_train, y_train, x_test, alpha, b, gamma)
        
        train_error = np.mean(train_predictions != y_train)
        test_error = np.mean(test_predictions != y_test)
        
        print(f"Training Error: {train_error:.4f}")
        print(f"Test Error: {test_error:.4f}")
        print("\n")
