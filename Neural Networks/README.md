
# Neural Network Algorithms

This repository contains Python implementations of neural network algorithms for different configurations and purposes.

## Usage Instructions

### 1. **File: `2a.py`**
#### Description
This file implements forward and backward passes for a neural network with two hidden layers. It includes an example dataset.

#### Steps to Execute
1. Ensure you have `numpy` and `pandas` installed in your environment.
2. Place your training data in a file named `train.csv` under the directory `bank-note/`.
3. Run the script directly:
   ```bash
   python 2a.py
   ```
   This will:
   - Load data from `bank-note/train.csv`.
   - Perform forward and backward passes for a single sample.
   - Output computed gradients for debugging.

---

### 2. **File: `2b.py`**
#### Description
This file trains a neural network using cross-entropy loss and backpropagation.

#### Steps to Execute
1. Ensure you have `numpy` and `matplotlib` installed.
2. Place your training and test data in `bank-note/train.csv` and `bank-note/test.csv`, respectively.
3. Execute the script:
   ```bash
   python 2b.py
   ```
   This will:
   - Train the neural network for multiple hidden layer widths.
   - Output train and test errors for each configuration.
   - Plot learning curves for each width.

---

### 3. **File: `2c.py`**
#### Description
This file modifies weight initialization to zeros and trains the neural network.

#### Steps to Execute
1. Follow the setup as for `2b.py` (same input data files and dependencies).
2. Run the script:
   ```bash
   python 2c.py
   ```
   This script performs the same process as `2b.py`, but initializes weights differently.

---

### 4. **File: `3.py`**
#### Description
This file trains a neural network using PyTorch, with configurable depth, width, and activation functions.

#### Steps to Execute
1. Install `numpy`, `torch`, and `scikit-learn` in your Python environment.
2. Preprocess your data as follows:
   - Place training and test data in `bank-note/train.csv` and `bank-note/test.csv`.
3. Execute the script:
   ```bash
   python 3.py
   ```
   This will:
   - Train models with different configurations (depth, width, activation).
   - Print training and testing errors for each configuration.

#### Key Parameters
- `depths`: Controls the number of hidden layers (e.g., 3, 5, 9).
- `widths`: Controls the number of units in each hidden layer (e.g., 5, 10, 25, 50, 100).
- `activations`: Choose between `'tanh'` or `'relu'`.
- Weight initialization:
  - `'tanh'` uses Xavier initialization.
  - `'relu'` uses He initialization.

---

For all files, ensure the `bank-note` directory contains the required CSV data files in the expected format before running the scripts.
