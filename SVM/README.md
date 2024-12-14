
# Support Vector Machine and Kernel Perceptron Algorithms

This repository contains Python implementations of various algorithms for SVM (Support Vector Machine) and Kernel Perceptron, including primal SVM, dual SVM with linear and Gaussian kernels, and the kernel perceptron.

## Requirements

- Python 3.6+
- Required libraries: `numpy`, `pandas`, `matplotlib`, `scipy`

Install the required libraries using:
```bash
pip install numpy pandas matplotlib scipy
```

## File Descriptions

1. **2a.py**: Implements SVM using stochastic gradient descent (SGD) for the primal optimization problem.
2. **2b.py**: A variant of SVM primal SGD with a different learning rate adjustment strategy.
3. **3a.py**: Dual SVM implementation for linear kernels.
4. **3b.py**: Dual SVM implementation with Gaussian kernels.
5. **3c.py**: Adds support vector tracking and kernel matrix overlap for Gaussian kernels.
6. **3d.py**: Implements Kernel Perceptron for binary classification.

## Usage

### Data Preparation
Ensure the training and test datasets are available at the following paths:
- Training data: `bank-note/train.csv`
- Testing data: `bank-note/test.csv`

Each file should contain features as columns and the last column as the label (`0` or `1`).

### Running the Scripts

#### 1. SVM Primal Optimization (2a.py, 2b.py)
Run the scripts to train SVM using SGD for the primal problem. Adjust parameters `C`, `gamma_0`, `a`, and `T` to control regularization, initial learning rate, learning rate decay, and iterations:
```bash
python 2a.py
python 2b.py
```

#### 2. SVM Dual Optimization with Linear Kernel (3a.py)
Run the script to solve the SVM dual problem using a linear kernel. Adjust `C` values in the script for regularization:
```bash
python 3a.py
```

#### 3. SVM Dual Optimization with Gaussian Kernel (3b.py, 3c.py)
Run the script to solve the SVM dual problem using a Gaussian kernel. Modify `C` and `gamma` values in the script:
```bash
python 3b.py
python 3c.py
```

#### 4. Kernel Perceptron (3d.py)
Run the script to train a kernel perceptron. Modify `gamma` and `max_iterations` in the script:
```bash
python 3d.py
```

## Parameters

- `C`: Regularization parameter for SVM.
- `gamma`: Parameter for the Gaussian kernel.
- `gamma_0`, `a`: Parameters for learning rate decay in primal SGD.
- `T`: Number of epochs in SGD.
- `max_iterations`: Number of iterations for kernel perceptron training.

## Outputs

Each script prints:
- Learned weights (`w`) and bias (`b`) for linear SVM models.
- Training and test errors.
- Additional metrics for kernel methods, such as support vector counts and overlaps (3c.py).

Modify the scripts to customize output or parameters as needed.
