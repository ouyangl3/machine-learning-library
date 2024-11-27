# Perceptron Algorithms

## Standard Perceptron
### 2a.py
This script implements the Standard Perceptron algorithm for binary classification. It trains a perceptron to classify samples from the `bank-note` dataset.

**How to Run**:
```
python 2a.py
```

**Parameters**:
- `epochs`: Number of iterations over the training dataset (default: 10).
- `r`: Learning rate, which controls the step size during weight update (default: 0.1).

**Dataset**: The training and test data should be located in `bank-note/train.csv` and `bank-note/test.csv`, respectively.

**Output**: The script prints the learned weights and bias, and also displays the average prediction error on the test data.

## Voted Perceptron
### 2b.py
This script implements the Voted Perceptron algorithm, which stores multiple weight vectors along with counts of how many times each vector was used.

**How to Run**:
```
python 2b.py
```

**Parameters**:
- `epochs`: Number of iterations over the training dataset (default: 10).
- `r`: Learning rate, which controls the step size during weight update (default: 0.1).

**Dataset**: The training and test data should be located in `bank-note/train.csv` and `bank-note/test.csv`, respectively.

**Output**: The script prints the weight vectors, biases, and counts for each vector, and also displays the average prediction error on the test data.

## Average Perceptron
### 2c.py
This script implements the Average Perceptron algorithm, which maintains a running average of all weight vectors during training for better generalization.

**How to Run**:
```
python 2c.py
```

**Parameters**:
- `epochs`: Number of iterations over the training dataset (default: 10).
- `r`: Learning rate, which controls the step size during weight update (default: 0.1).

**Dataset**: The training and test data should be located in `bank-note/train.csv` and `bank-note/test.csv`, respectively.

**Output**: The script prints the learned average weights and bias, and also displays the average prediction error on the test data.

