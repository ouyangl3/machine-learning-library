import csv
import math
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def vector_norm(v):
    return sum(abs(v_i) for v_i in v)

def stochastic_gradient_descent(x, y, weights, learning_rate, tolerance=1e-6):
    m = len(y)
    cost_history = []
    while True:
        random_sample = np.random.randint(m)
        x_i = x[random_sample, :].reshape(1, -1)
        y_i = y[random_sample].reshape(-1)
        
        prediction = np.dot(x_i, weights)
        error = prediction - y_i
        
        gradient = np.dot(x_i.T, error) / x_i.shape[0]
        
        weights -= learning_rate * gradient
        
        cost = np.sum((np.dot(x, weights) - y) ** 2) / 2
        cost_history.append(cost)
        
        if len(cost_history) > 1 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            return weights, cost_history
    
def calculate_cost(x, y, weights):
    m = len(y)
    predictions = x.dot(weights)
    errors = predictions - y
    cost = np.sum(errors ** 2) / 2 
    return cost

train_data = pd.read_csv('concrete/concrete/train.csv')
test_data = pd.read_csv('concrete/concrete/test.csv')

# Extracting features and labels
train_x = train_data.iloc[:, :-1].values
train_y = train_data.iloc[:, -1].values
test_x = test_data.iloc[:, :-1].values
test_y = test_data.iloc[:, -1].values

# Initialize weights and learning rate
lr = 0.001
weights = np.zeros(train_x.shape[1])

weights, costs = stochastic_gradient_descent(train_x, train_y, weights, lr)

print("The learned weight vector:", weights)
test_cost = calculate_cost(test_x, test_y, weights)
print("The cost function value of the test data:", test_cost)

plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()