import numpy as np
import pandas as pd

def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))

def sigmoid_deriv(z):
    return z * (1.0 - z)

def forward_pass(X, W1, W2, W3):
    # Add bias to input
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    s1 = Xb @ W1.T
    z1 = sigmoid(s1)
    z1b = np.hstack([np.ones((X.shape[0], 1)), z1])
    s2 = z1b @ W2.T
    z2 = sigmoid(s2)
    z2b = np.hstack([np.ones((X.shape[0], 1)), z2])
    s3 = z2b @ W3
    y_hat = sigmoid(s3)
    return Xb, s1, z1, z1b, s2, z2, z2b, s3, y_hat

def backward_pass(Xb, s1, z1, z1b, s2, z2, z2b, s3, y_hat, y, W1, W2, W3):
    dL_ds3 = (y_hat - y.reshape(-1, 1))
    dL_dW3 = (z2b.T @ dL_ds3) / Xb.shape[0]

    dL_dz2 = dL_ds3 @ W3[1:].T
    dL_ds2 = dL_dz2 * z2 * (1 - z2)
    dL_dW2 = (z1b.T @ dL_ds2) / Xb.shape[0]

    dL_dz1 = dL_ds2 @ W2[:, 1:]
    dL_ds1 = dL_dz1 * z1 * (1 - z1)
    dL_dW1 = (Xb.T @ dL_ds1) / Xb.shape[0]

    return dL_dW1.T, dL_dW2.T, dL_dW3

def initialize_weights(input_dim, hidden1, hidden2, output_dim):
    W1 = np.random.randn(hidden1, input_dim + 1)
    W2 = np.random.randn(hidden2, hidden1 + 1)
    W3 = np.random.randn(hidden2 + 1, output_dim)
    return W1, W2, W3

if __name__ == "__main__":
    # Example dataset
    data = pd.read_csv('bank-note/train.csv', header=None).values
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Labels

    # Convert labels to correct shape
    y = y.reshape(-1, 1)

    # Use one sample for testing
    X_single = X[0:1, :]
    y_single = y[0:1]

    hidden1 = 5  # Number of units in hidden layer 1
    hidden2 = 3  # Number of units in hidden layer 2

    # Initialize weights
    W1, W2, W3 = initialize_weights(X.shape[1], hidden1, hidden2, 1)

    # Perform forward and backward pass for one sample
    Xb, s1, z1, z1b, s2, z2, z2b, s3, y_hat = forward_pass(X_single, W1, W2, W3)
    dL_dW1, dL_dW2, dL_dW3 = backward_pass(Xb, s1, z1, z1b, s2, z2, z2b, s3, y_hat, y_single, W1, W2, W3)

    # Print gradients
    print("Gradients:")
    print(f"dL/dW1:\n{dL_dW1}")
    print(f"dL/dW2:\n{dL_dW2}")
    print(f"dL/dW3:\n{dL_dW3}")

