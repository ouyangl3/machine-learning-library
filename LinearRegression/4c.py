import csv
import math
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def vector_norm(v):
    return sum(abs(v_i) for v_i in v)

def normal_equation(x, y):
    x_transpose = x.T
    weights = np.linalg.inv(x_transpose.dot(x)).dot(x.T).dot(y)
    return weights
    
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

weights = np.zeros(train_x.shape[1])

analytical_weights = normal_equation(train_x, train_y)

print("analytical_weights:", analytical_weights)