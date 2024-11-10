# AdaBoost
2a.py
    The script 2a.py implements an AdaBoost algorithm based on decision trees. The code includes the following main components:

        Function weighted_error_rate: Calculates the weighted error rate of the predictions to assess the accuracy of the current model.

        Function draw: Plots the training and testing error rates for AdaBoost and decision trees, showing their changes with each iteration.

        Function adaboost: Implements the AdaBoost algorithm. In each iteration, it generates a weighted decision stump and updates the sample weights.

        Function numerical_to_binary: Binarizes numerical features (e.g., sets values above the median to 1, below the median to 0).

        Decision tree-related functions:
            calculate_entropy, calculate_majority_error, calculate_gini: Calculate different split criteria.
            id3: Implements the ID3 algorithm, which splits the data based on different features to build a decision tree.
            predict: Predicts the label for new samples.

        Main Program: Reads training and testing data, uses the adaboost function to train the model, and calls the draw function to plot the training and testing error curves.

    Implemented Ensemble Learning Method
        This code implements the AdaBoost algorithm, which improves classification performance by using multiple weak classifiers (decision stumps) and weighting updates based on the error rate.


    File Structure
        Input Data: Reads and preprocesses data from files.
        AdaBoost Algorithm: Uses the ID3 algorithm to generate stumps and progressively optimize the model.
        Results Visualization: Plots error curves to visualize the difference in training and testing performance.

# Bagging
2b.py
    The script 2b.py implements a Bagging (Bootstrap Aggregation) algorithm based on decision trees. The main components of the code are as follows:

        Function bagged_decision_trees: Implements the Bagging algorithm. This function generates multiple data subsets through bootstrapping (sampling with replacement) and trains a decision tree model on each subset. All decision tree models form a Bagging model.

        Function predict_bagging: Uses multiple decision tree models to make predictions on samples. It performs voting across all trees’ predictions to determine the final prediction of the Bagging model.

        Function numerical_to_binary: Binarizes numerical features in the dataset (e.g., sets values above the median to 1 and below the median to 0) and converts "yes" in the label column to 1 and "no" to -1.

        Decision tree helper functions:
            calculate_entropy, calculate_majority_error, calculate_gini: Used for different split criteria (entropy, majority error rate, Gini index).
            calculate_measure: Calculates the purity of samples based on the specified criterion.
            best_attribute: Selects the best feature for splitting to maximize information gain.
            id3: Implements the ID3 algorithm to generate a single decision tree.
            predict: Makes predictions for samples based on the decision tree model.
            calculate_error_rate: Calculates the error rate of model predictions.

        Function plot_error_rates: Plots the error rate curves of the Bagging model on the training and testing sets, showing how error rates change with an increasing number of trees.

        Main Program: The main program reads training and testing data from files, converts the data to binary form, and trains the model using the Bagging algorithm. Finally, it plots the training and testing error curves.

    Implemented Ensemble Learning Method
        This file implements the Bagging algorithm, which reduces the variance of individual models by combining randomly sampled data and decision trees, enhancing the overall model’s stability and accuracy.

    File Structure
        Data Preprocessing: Binarizes numerical features in the dataset to facilitate processing by the algorithm.
        Bagging Algorithm: Trains multiple decision trees and uses a voting mechanism to obtain the final prediction.
        Results Visualization: Plots training and testing error curves to demonstrate model performance.

2c.py
    The script 2c.py implements a Bagging algorithm based on decision trees, similar to 2b.py, but with the addition of a Bias-Variance analysis. The main components of the code include:

        Function bagged_decision_trees: Implements the Bagging algorithm. It generates multiple data subsets through bootstrapping (sampling with replacement) and trains a decision tree model on each subset. All decision tree models together form the Bagging model.

        Function predict_bagging: Uses the Bagging model to make predictions on samples. It performs voting across all trees’ predictions to determine the final prediction result.

        Function numerical_to_binary: Binarizes numerical features in the dataset (e.g., sets values above the median to 1 and below the median to 0) and converts "yes" in the label column to 1 and "no" to -1.

        Decision tree helper functions:
            calculate_entropy, calculate_majority_error, calculate_gini: Calculate different split criteria.
            calculate_measure: Calculates the purity of samples based on the specified criterion (entropy, error rate, or Gini index).
            best_attribute: Selects the best feature for splitting to maximize information gain.
            id3: Implements the ID3 algorithm to generate a single decision tree.
            predict: Uses the decision tree model to make predictions on samples.
            calculate_error_rate: Calculates the error rate of the model.

        Function plot_error_rates: Plots the error rate curves of the Bagging model on the training and testing sets, showing how error rates change as the number of decision trees increases.

        Function calculate_bias_variance_error: Calculates the bias-variance decomposition error. It calculates bias and variance as well as the overall mean squared error using the predictions from multiple individual decision trees. It consists of two parts:
            Bias: The deviation between the mean of the predictions and the true labels.
            Variance: The variance of the predictions, reflecting the model's sensitivity to data fluctuations.

        Main Program
            Reads and binarizes the training and testing data.
            Trains multiple Bagging models and performs bias-variance decomposition analysis.
            Calculates and outputs the bias, variance, and overall mean squared error for a single decision tree and the Bagging model.

    Implemented Ensemble Learning Method
        This file also implements the Bagging algorithm. Unlike 2b.py, it adds bias-variance analysis, which helps in understanding the model's generalization ability, especially in balancing bias (underfitting) and variance (overfitting).

    File Structure
        Data Preprocessing: Binarizes numerical features in the dataset to facilitate processing by the algorithm.
        Bagging Algorithm: Trains multiple decision trees and uses a voting mechanism to obtain the final prediction.
        Bias-Variance Analysis: Calculates the bias, variance, and overall mean squared error of a single tree and the Bagging model, analyzing the model’s performance.
        Results Visualization: Outputs the model’s bias, variance, and overall mean squared error, and plots error curves.

# Random Forest
2d.py
    The script 2d.py implements a Random Forest algorithm based on decision trees and provides a Bias-Variance analysis. Similar to the previous Bagging implementation, but in this file, a random subset of features is chosen at each split during the training of each decision tree, further reducing overfitting. The main components of the code include:

        Function random_forest: Implements the Random Forest algorithm. It generates multiple data subsets through bootstrapping (sampling with replacement) and trains a decision tree on each subset. Each decision tree splits based on a random subset of features, further reducing overfitting.

        Function bagged_decision_trees: Same as previous implementations, generating multiple decision tree models based on bootstrapping without random feature selection. This function is a simple Bagging model.

        Function predict_bagging: Uses either the Bagging or Random Forest model to make predictions on samples. It performs voting across all trees’ predictions to determine the final prediction result.

        Function numerical_to_binary: Binarizes numerical features in the dataset (e.g., sets values above the median to 1, below the median to 0) and converts "yes" in the label column to 1 and "no" to -1.

        Decision tree helper functions:
            calculate_entropy, calculate_majority_error, calculate_gini: Calculate different split criteria.
            calculate_measure: Calculates the purity of samples based on the specified criterion (entropy, error rate, or Gini index).
            best_attribute: Selects the best feature for splitting to maximize information gain.
            id3: Implements the ID3 algorithm to generate a single decision tree, supporting random feature selection for splitting.
            predict: Uses the decision tree model to make predictions on samples.
            calculate_error_rate: Calculates the error rate of the model.

        Function plot_error_rates: Plots the error rate curves of the Random Forest model on the training and testing sets, showing how error rates change with an increasing number of trees and comparing the effects of different feature subset sizes.

        Bias-Variance Analysis
            Function calculate_bias_variance_error: Calculates bias and variance. Using predictions from multiple individual decision trees, it calculates bias, variance, and the overall Mean Squared Error (MSE) in the following parts:
                Bias: The deviation between the average prediction and the true labels.
                Variance: The variance of the prediction results, reflecting the model's sensitivity to data fluctuations.
                Overall Mean Squared Error: The sum of bias and variance.

        Main Program
            Reads and binarizes the training and testing data.
            Trains multiple models using the Random Forest algorithm, performing bias-variance decomposition analysis.
            Analyzes the impact of different feature subset sizes on model performance, and visualizes the change in error rate with the number of trees and feature subset sizes using the plot_error_rates function.

    Implemented Ensemble Learning Method
        This file implements the Random Forest algorithm. Compared to Bagging, Random Forest introduces randomness in feature selection during tree construction to reduce model variance and enhance generalization ability. Additionally, the file includes bias-variance analysis to gain a deeper understanding of model performance.

    File Structure
        Data Preprocessing: Binarizes numerical features in the dataset.
        Random Forest Algorithm: Builds multiple decision trees through data and feature sampling.
        Bias-Variance Analysis: Calculates bias, variance, and overall mean squared error for a single tree and the Random Forest model, analyzing model generalization ability.
        Results Visualization: Outputs bias, variance, and overall mean squared error, and plots error rate curves, analyzing the effect of feature subset size on the error rate.

2e.py
    The script 2e.py implements a Random Forest algorithm based on decision trees, similar to 2d.py, but with an additional focus on the impact of different feature subset sizes on the performance of the Random Forest. The main components of the code include:

        Function random_forest: Implements the Random Forest algorithm. It generates multiple data subsets through bootstrapping (sampling with replacement) and trains a decision tree on each subset. Each decision tree splits based on a random subset of features, helping to reduce overfitting.

        Function bagged_decision_trees: Implements the Bagging algorithm. It generates multiple data subsets through bootstrapping and trains a decision tree model on each subset, forming a Bagging model.

        Function predict_bagging: Uses either the Bagging or Random Forest model to make predictions on samples. It performs voting across all trees' predictions to determine the final prediction result.

        Function numerical_to_binary: Binarizes numerical features in the dataset (e.g., sets values above the median to 1 and below the median to 0) and converts "yes" in the label column to 1 and "no" to -1.

        Decision tree helper functions:
            calculate_entropy, calculate_majority_error, calculate_gini: Calculate different split criteria.
            calculate_measure: Calculates the purity of samples based on the specified criterion (entropy, error rate, or Gini index).
            best_attribute: Selects the best feature for splitting to maximize information gain.
            id3: Implements the ID3 algorithm to generate a single decision tree, supporting random feature selection for splitting.
            predict: Uses the decision tree model to make predictions on samples.
            calculate_error_rate: Calculates the error rate of the model.

        Function plot_error_rates: Plots error rate curves for Random Forest models with different feature subset sizes on the training and testing sets. By comparing different feature subset sizes, it shows their impact on model performance.

        Bias-Variance Analysis
            Function calculate_bias_variance_error: Calculates bias and variance. Using predictions from a single decision tree and multiple predictions from the Random Forest, it calculates bias, variance, and overall Mean Squared Error (MSE), specifically including:
                Bias: The deviation between the average prediction and the true labels.
                Variance: The variance of the prediction results, reflecting the model's sensitivity to data fluctuations.
                Overall Mean Squared Error: The sum of bias and variance.

        Main Program
            Reads and binarizes the training and testing data.
            Trains multiple models using the Random Forest algorithm and performs bias-variance decomposition analysis.
            Analyzes the impact of different feature subset sizes on the model's bias and variance.
            Uses the plot_error_rates function to visualize the changes in error rate with the number of trees and feature subset sizes.

    Implemented Ensemble Learning Method
        This file implements the Random Forest algorithm and compares the effects of different feature subset sizes in the bias-variance analysis. By introducing feature randomness during construction, Random Forest reduces model variance and enhances generalization ability. Additionally, bias-variance analysis provides a deeper understanding of model performance, especially in balancing bias and variance.

    File Structure
        Data Preprocessing: Binarizes numerical features in the dataset.
        Random Forest Algorithm: Builds multiple decision trees through data and feature sampling.
        Bias-Variance Analysis: Calculates bias, variance, and overall mean squared error for a single tree and the Random Forest model, analyzing the model's generalization ability.
        Results Visualization: Outputs bias, variance, and overall mean squared error and plots error rate curves to analyze the effect of feature subset size on the error rate.
