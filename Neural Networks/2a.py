import numpy as np

def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))

def sigmoid_deriv(z):
    return z*(1.0 - z)

x = np.array([1.0, 1.0, 1.0])
y_star = 1.0

W1 = np.array([
    [-1.0, -2.0, -3.0],
    [ 1.0,  2.0,  3.0]
])

W2 = np.array([
    [-1.0, -2.0, -3.0],
    [ 1.0,  2.0,  3.0]
])

W3 = np.array([
    [-1.0],
    [ 2.0],
    [-1.5]
])

x_input = np.array([1.0, 1.0, 1.0])
s1 = W1 @ x_input
z1 = sigmoid(s1)
z1_with_bias = np.concatenate(([1.0], z1))
s2 = W2 @ z1_with_bias
z2 = sigmoid(s2)
z2_with_bias = np.concatenate(([1.0], z2))
s3 = W3.T @ z2_with_bias
y = sigmoid(s3[0])
L = 0.5*(y - y_star)**2
dL_dy = (y - y_star)
dL_ds3 = dL_dy * y*(1-y)
dL_dW3 = (dL_ds3 * z2_with_bias).reshape(-1,1)
dL_dz2 = (W3[1:,0] * dL_ds3)
dL_ds2 = dL_dz2 * z2*(1-z2)
dL_dW2 = np.outer(dL_ds2, z1_with_bias)
dL_dz1 = (W2[:,1:].T @ dL_ds2)
dL_ds1 = dL_dz1 * z1*(1-z1)
dL_dW1 = np.outer(dL_ds1, x_input)

print("y =", y)
print("Loss L =", L)

print("\nGradients:")
print("dL/dW3:\n", dL_dW3)
print("dL/dW2:\n", dL_dW2)
print("dL/dW1:\n", dL_dW1)
