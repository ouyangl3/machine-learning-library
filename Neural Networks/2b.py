import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def forward_pass(X, W1, W2, W3):
    # X shape: (N, d)
    # Add bias to input
    N = X.shape[0]
    Xb = np.hstack([np.ones((N,1)), X])
    s1 = Xb @ W1.T    # (N, hidden1)
    z1 = sigmoid(s1)
    z1b = np.hstack([np.ones((N,1)), z1])
    s2 = z1b @ W2.T   # (N, hidden2)
    z2 = sigmoid(s2)
    z2b = np.hstack([np.ones((N,1)), z2])
    s3 = z2b @ W3     # (N, 1)
    y_hat = sigmoid(s3)
    return Xb, s1, z1, z1b, s2, z2, z2b, s3, y_hat

def backward_pass(Xb, s1, z1, z1b, s2, z2, z2b, s3, y_hat, y, W1, W2, W3):
    N = Xb.shape[0]
    # Cross-entropy loss: L = -1/N sum [y*log(y_hat)+(1-y)*log(1-y_hat)]
    # dL/dy_hat = (y_hat - y)/(y_hat*(1-y_hat)) for cross entropy with sigmoid
    # However, a stable form: dL/ds3 = y_hat - y
    # Because for cross-entropy and sigmoid: d/ds3 = y_hat - y
    dL_ds3 = (y_hat - y.reshape(-1,1))  # (N,1)

    # W3 shape: (hidden2+1, 1)
    dL_dW3 = (z2b[:,:,None] * dL_ds3[:,None,:]).sum(axis=0) # sum over N
    dL_dW3 = dL_dW3.reshape(-1,1)

    # Backprop to layer 2
    # dL/dz2 = dL/ds3 * W3(1:)
    # W3[1:] corresponds to non-bias weights
    dL_dz2 = dL_ds3 @ W3[1:].T  # (N, hidden2)
    dL_ds2 = dL_dz2 * z2*(1-z2) # (N, hidden2)

    # dL/dW2
    # z1b shape: (N, hidden1+1)
    dL_dW2 = np.einsum('nk, nj->kj', dL_ds2, z1b)

    # Backprop to layer 1
    # dL/dz1 = dL/ds2 * W2(1:)
    dL_dz1 = dL_ds2 @ W2[:,1:]  # (N,hidden1)
    dL_ds1 = dL_dz1 * z1*(1-z1) # (N,hidden1)

    dL_dW1 = np.einsum('nk, nj->kj', dL_ds1, Xb)

    return dL_dW1, dL_dW2, dL_dW3

def predict(X, W1, W2, W3):
    _, _, _, _, _, _, _, _, y_hat = forward_pass(X, W1, W2, W3)
    return (y_hat > 0.5).astype(int)

def compute_error(y_pred, y):
    return np.mean(y_pred.reshape(-1) != y)

def compute_cross_entropy_loss(y_hat, y):
    # y_hat (N,1), y (N,)
    y = y.reshape(-1,1)
    # Avoid log(0)
    eps = 1e-12
    return -np.mean(y*np.log(y_hat+eps)+(1-y)*np.log(1-y_hat+eps))

def train_neural_network(X_train, y_train, X_test, y_test, hidden_width=10, epochs=100, gamma_0=0.1, d=1.0):
    input_dim = X_train.shape[1]
    np.random.seed(42)
    # Initialize weights
    # Layer1: hidden_width units, input_dim+1 inputs
    W1 = np.random.randn(hidden_width, input_dim+1)
    # Layer2: hidden_width units, hidden_width+1 inputs
    W2 = np.random.randn(hidden_width, hidden_width+1)
    # Layer3: 1 unit, hidden_width+1 inputs
    W3 = np.random.randn(hidden_width+1, 1)

    t = 0
    N = X_train.shape[0]

    loss_history = []

    for epoch in range(epochs):
        # Shuffle training data
        idx = np.arange(N)
        np.random.shuffle(idx)
        X_train = X_train[idx]
        y_train = y_train[idx]

        for i in range(N):
            x_i = X_train[i:i+1]
            y_i = y_train[i:i+1]
            Xb, s1, z1, z1b, s2, z2, z2b, s3, y_hat = forward_pass(x_i, W1, W2, W3)

            loss = compute_cross_entropy_loss(y_hat, y_i)
            loss_history.append(loss)

            dL_dW1, dL_dW2, dL_dW3 = backward_pass(Xb, s1, z1, z1b, s2, z2, z2b, s3, y_hat, y_i, W1, W2, W3)
            t += 1
            gamma_t = gamma_0/(1+(gamma_0/d)*t)
            W1 -= gamma_t*dL_dW1
            W2 -= gamma_t*dL_dW2
            W3 -= gamma_t*dL_dW3

    # After training, compute training and test error
    y_pred_train = predict(X_train, W1, W2, W3)
    y_pred_test = predict(X_test, W1, W2, W3)
    train_error = compute_error(y_pred_train, y_train)
    test_error = compute_error(y_pred_test, y_test)
    return train_error, test_error, loss_history

if __name__ == "__main__":
    X_train, y_train = load_data("bank-note/train.csv")
    X_test, y_test = load_data("bank-note/test.csv")

    widths = [5, 10, 25, 50, 100]
    for w in widths:
        train_err, test_err, loss_history = train_neural_network(X_train, y_train, X_test, y_test, hidden_width=w, epochs=100, gamma_0=0.1, d=1.0)
        print(f"Width={w}: Train Error={train_err:.4f}, Test Error={test_err:.4f}")

        plt.figure()
        plt.plot(loss_history)
        plt.title(f'Learning Curve (Width={w})')
        plt.xlabel('Update Iteration')
        plt.ylabel('Cross-Entropy Loss')
        plt.grid(True)
        plt.show()