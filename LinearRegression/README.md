# Batch Gradient Descent
4a.py
    The script 4a.py implements a Batch Gradient Descent algorithm to train a linear regression model for predicting concrete strength. The main components of the code include:

        Function vector_norm: Calculates the L1 norm of a vector, which is the sum of the absolute values of all elements, used to assess the convergence of gradient descent.

        Function batch_gradient_descent: Implements the Batch Gradient Descent algorithm.
            Input: Training data x and y, initial weights weights, learning rate learning_rate, and tolerance for convergence tolerance.
            Algorithm Process: In each iteration, it calculates the predictions and errors, computes the gradient based on the error, and updates the weights. After each iteration, it records the cost (loss) value. If the change in weights is smaller than the tolerance, the algorithm stops and returns the final weights and cost history.
            
        Function calculate_cost: Calculates the cost (loss) value for the current weights, evaluating the model's performance on test data.

        Main Program
            Reads the training and testing data from the CSV files train.csv and test.csv.
            Initializes the learning rate and weights and calls the batch_gradient_descent function for training.
            Prints the weight vector obtained from training and calculates the cost on the test data.
            Plots the cost history graph, showing the trend of cost change over the iterations.

    Implemented Linear Regression Method
        This file uses batch gradient descent to optimize a linear regression model, which is suitable for handling larger batches of data. Batch gradient descent calculates the gradient using all samples in the training set for each iteration, resulting in a more stable update direction.

    File Structure
        Data Preprocessing: Reads training and testing data from CSV files.
        Batch Gradient Descent Algorithm: Uses batch gradient descent to optimize the weights of the linear regression model.
        Results Visualization: Prints the weight vector obtained from training, calculates the cost on the test set, and plots the cost change with iterations.

# Stochastic Gradient Descent
4b.py
    The script 4b.py implements a Stochastic Gradient Descent (SGD) algorithm to train a linear regression model for predicting concrete strength. Compared to Batch Gradient Descent, SGD updates the weights using only one sample at a time, resulting in higher update frequency and faster convergence, though with a more oscillatory convergence path. The main components of the code include:

        Function vector_norm: Calculates the L1 norm of a vector, which is the sum of the absolute values of all elements, used to assess the convergence of gradient descent.

        Function stochastic_gradient_descent: Implements the Stochastic Gradient Descent algorithm.
            Input: Training data x and y, initial weights weights, learning rate learning_rate, and tolerance for convergence tolerance.
            Algorithm Process: Each time, it randomly selects one sample, calculates the prediction and error for that sample, computes the gradient for that sample based on the error, and updates the weights. After each iteration, it records the cost (loss) value. When the change in cost is less than the tolerance, the iterations stop, returning the final weights and cost history.
        
        Function calculate_cost: Calculates the cost (loss) value for the current weights, used to evaluate the model’s performance on the test data.

        Main Program
            Reads the training and testing data from the CSV files train.csv and test.csv.
            Initializes the learning rate and weights and calls the stochastic_gradient_descent function for training.
            Prints the weight vector obtained from training and calculates the cost on the test data.
            Plots the cost history graph, showing the trend of cost change over the iterations.

    Implemented Linear Regression Method
        This file uses stochastic gradient descent to optimize the linear regression model, making it suitable for large datasets. Since SGD updates the weights using one random sample per iteration, the computational cost is low, making it ideal for real-time parameter updates. While the convergence path is more oscillatory, it converges faster with an appropriate learning rate.

    File Structure
        Data Preprocessing: Reads training and testing data from CSV files.
        Stochastic Gradient Descent Algorithm: Optimizes the weights of the linear regression model using stochastic gradient descent.
        Results Visualization: Prints the weight vector obtained from training, calculates the cost on the test set, and plots the cost change with iterations.

# Normal Equation
4c.py
    The script 4c.py implements the Normal Equation method for solving the weights of a linear regression model. The Normal Equation is a direct method for calculating the parameters of a linear regression model through matrix operations, without the need for iterative optimization. The main components of the code include:

        Function vector_norm: Calculates the L1 norm of a vector, which is the sum of the absolute values of all elements.

        Function normal_equation: Implements the Normal Equation method to compute the linear regression weights.
            Input: Feature matrix x and target vector y.
            Process: First, it computes the transpose of x, then calculates the weights using the formula weights = (X^T * X)^(-1) * X^T * y. This computation is done directly through matrix operations, so no learning rate or iteration is needed.
        
        Function calculate_cost: Calculates the cost (loss) value for the current weights, used to evaluate the model’s performance on the test data.

        Main Program
            Reads the training and testing data from the CSV files train.csv and test.csv.
            Extracts features and labels, calculates the weights analytical_weights using the Normal Equation.
            Prints the calculated weight vector analytical_weights.

    Implemented Linear Regression Method
        This file uses the Normal Equation to solve for the weights of the linear regression model. The Normal Equation is a direct method that calculates the optimal solution in one step using matrix computation, making it suitable for small datasets. Its advantage is that it does not require a learning rate or iterations. However, when the number of features is large, computing the inverse of X^T * X can be expensive or even infeasible.

    File Structure
        Data Preprocessing: Reads training and testing data from CSV files.
        Normal Equation: Directly solves for the weights of the linear regression model using the Normal Equation.
        Results Display: Prints the calculated weight vector.
