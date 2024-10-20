import csv
import math
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def vector_norm(v):
    return sum(abs(v_i) for v_i in v)

def batch_gradient_descent(x, y, weights, learning_rate, tolerance=1e-6):
    m = len(y)
    cost_history = []
    
    while True:
        predictions = x.dot(weights)
        errors = predictions - y
        
        # calculate gradients
        gradients = np.dot(x.T, errors)

        # update new_weights
        new_weights = weights - learning_rate * gradients

        cost = np.sum((errors) ** 2) / 2
        cost_history.append(cost)
        
        # convergence
        if vector_norm(new_weights - weights) < tolerance:
            return new_weights, cost_history

        weights = new_weights
    
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
lr = 0.01
weights = np.zeros(train_x.shape[1])

weights, costs = batch_gradient_descent(train_x, train_y, weights, lr)

print("The learned weight vector:", weights)
test_cost = calculate_cost(test_x, test_y, weights)
print("The cost function value of the test data:", test_cost)

plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()