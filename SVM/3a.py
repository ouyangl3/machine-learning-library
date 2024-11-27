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

def svm_dual_svm(X, y, C):
    n_samples = X.shape[0]
    K = np.dot(X, X.T)
    
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
    
    w = np.dot((alpha * y), X)
    
    support_vector_indices = np.where((alpha > 1e-5) & (alpha < C - 1e-5))[0]
    if len(support_vector_indices) == 0:
        support_vector_indices = np.where(alpha > 1e-5)[0]
    b = np.mean([y[i] - np.dot(w, X[i]) for i in support_vector_indices])
    
    return w, b, alpha

train_path = "bank-note/train.csv"
test_path = "bank-note/test.csv"

x_train, y_train, x_test, y_test = load_data(train_path, test_path)

C_values = [100/873, 500/873, 700/873]
gamma_0 = 0.5
a = 1
T = 100

for C in C_values:
    print(f"Training with C = {C:.4f}")
    
    w_dual, b_dual, alpha = svm_dual_svm(x_train, y_train, C)
    print(f"Weights (dual): {w_dual}")
    print(f"Bias (dual): {b_dual}")
    
    train_predictions_dual = np.sign(np.dot(x_train, w_dual) + b_dual)
    test_predictions_dual = np.sign(np.dot(x_test, w_dual) + b_dual)
    
    train_error_dual = np.mean(train_predictions_dual != y_train)
    test_error_dual = np.mean(test_predictions_dual != y_test)
    
    print(f"Training Error (dual): {train_error_dual:.4f}")
    print(f"Test Error (dual): {test_error_dual:.4f}")
    print("\n")
