import csv
import math
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
import pandas as pd
import matplotlib.pyplot as plt
import random

def random_forest(examples, attributes, attribute_names, T, feature_subset_size):
    # sample_size = len(examples)
    sample_size = 1000
    trees = []

    for t in range(T):
        bootstrap_sample = resample(examples, n_samples=sample_size, replace=True)
        tree = id3(bootstrap_sample, attributes, attribute_names, 'entropy', max_depth=None, feature_subset_size=feature_subset_size)
        trees.append(tree)

    return trees


def bagged_decision_trees(examples, attributes, attribute_names, T):
    # sample_size = len(examples)
    sample_size = 1000
    trees = []
    for t in range(T):
        bootstrap_sample = resample(examples, n_samples=sample_size, replace=True)
        tree = id3(bootstrap_sample, attributes, attribute_names, 'entropy', max_depth=None)
        trees.append(tree)

    return trees

def predict_bagging(trees, examples, attribute_indices):
    all_predictions = []
    for example in examples:
        predictions = [predict(tree, example, attribute_indices) for tree in trees]
        all_predictions.append(predictions)

    df = pd.DataFrame(all_predictions)
    final_predictions = df.mode(axis=1)[0]
    # print(f"Number of predictions: {len(final_predictions)}")
    # print(f"Number of actual examples: {len(examples)}") 
    
    return final_predictions


def numerical_to_binary(filepath, attribute_indices):
    data = pd.read_csv(filepath, header=None)
    numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    for feature in numerical_features:
        column_name = data.columns[attribute_indices[feature]]
        median_value = data[column_name].median()

        # 1 if above median, 0 otherwise
        data[column_name] = np.where(data[column_name].astype(float) > median_value, 1, 0)

    label_column = data.columns[-1]
    data[label_column] = data[label_column].apply(lambda x: 1 if x == 'yes' else -1)

    return data.values.tolist()

def majority_label(examples, label_column):
    labels = [example[label_column] for example in examples]

    # Return the most common label
    return Counter(labels).most_common(1)[0][0]

def calculate_entropy(examples, label_column):
    each_label_count = Counter(example[label_column] for example in examples)
    total = len(examples)
    # Avoid division by zero
    if total == 0: 
        return 0

    entropy = -sum((count / total) * math.log2(count / total) for count in each_label_count.values())
    return entropy

def calculate_majority_error(examples, label_column):
    each_label_count = Counter(example[label_column] for example in examples)
    total = len(examples)
    if total == 0: 
        return 0

    majority = each_label_count.most_common(1)[0][1]
    majority_error = 1 - (majority / total)
    return majority_error

def calculate_gini(examples, label_column):
    each_label_count = Counter(example[label_column] for example in examples)
    total = len(examples)
    if total == 0: 
        return 0

    gini = 1 - sum((count/total)**2 for count in each_label_count.values())
    return gini

def calculate_measure(examples, method, label_column):
    if method == 'entropy':
        return calculate_entropy(examples, label_column)
    elif method == 'majority_error':
        return calculate_majority_error(examples, label_column)
    elif method == 'gini':
        return calculate_gini(examples, label_column)

    raise 'The method is invalid'

def best_attribute(examples, attributes, method, label_column):
    first_measure = calculate_measure(examples, method, label_column)

    best_gain = -1
    best_attribute = None

    for attribute in attributes:
        subsets = defaultdict(list)
        for example in examples:
            subsets[example[attribute]].append(example)

        total = len(examples)
        expected_measure = sum((len(subset) / total) * calculate_measure(subset, method, label_column) for subset in subsets.values())

        gain = first_measure - expected_measure
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute

    # Return the attribute that has the greatest gain
    return best_attribute

def get_all_possible_values_for_attribute(attribute_name):
    attribute_values = {
        'age': [0, 1],
        'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student',
                'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
        'marital': ['married', 'divorced', 'single'],
        'education': ['unknown', 'secondary', 'primary', 'tertiary'],
        'default': ['yes', 'no'],
        'balance': [0, 1],
        'housing': ['yes', 'no'],
        'loan': ['yes', 'no'],
        'contact': ['unknown', 'telephone', 'cellular'],
        'day': [0, 1],
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'duration': [0, 1],
        'campaign': [0, 1],
        'pdays': [0, 1],
        'previous': [0, 1],
        'poutcome': ['unknown', 'other', 'failure', 'success']
    }

    return attribute_values.get(attribute_name, [])

def id3(examples, attributes, attribute_names, method, max_depth, current_depth=0, feature_subset_size=None):
    label_column = -1
    # If all examples have same label, return a leaf node with the label
    if len(set(example[label_column] for example in examples)) == 1:
        return examples[0][label_column]

    # If Attributes is empty or if the maximum depth has been reached, return a leaf node with the majority label
    if not attributes or current_depth == max_depth:
        return majority_label(examples, label_column)

    if feature_subset_size:
        available_attributes = random.sample(attributes, min(len(attributes), feature_subset_size))
    else:
        available_attributes = list(attributes)

    best_attribute_index = best_attribute(examples, available_attributes, method, label_column)
    best_attribute_name = attribute_names[best_attribute_index]

    tree = {best_attribute_name: {}}

    all_possible_values = get_all_possible_values_for_attribute(best_attribute_name)

    for value in all_possible_values:
        subset = [example for example in examples if example[best_attribute_index] == value]

        # For each value, recurse to create subtrees
        if subset:
            # Remove the best attribute for further splits
            new_attributes = attributes - {best_attribute_index}
            tree[best_attribute_name][value] = id3(subset, new_attributes, attribute_names, method, max_depth, current_depth + 1, feature_subset_size)
        else:
            # If no examples have this value, use the majority label
            tree[best_attribute_name][value] = majority_label(examples, label_column)

    return tree

def predict(tree, example, attribute_indices):
    # Base case: if the current node is a leaf node, return its value
    if not isinstance(tree, dict):
        return tree

    # Get the first key in the dictionary as the attribute
    attribute = next(iter(tree))
    attribute_value = example[attribute_indices[attribute]]
    
    # Check if the attribute value has a corresponding subtree
    if attribute_value in tree[attribute]:
        subtree = tree[attribute][attribute_value]
    
    return predict(subtree, example, attribute_indices)

def calculate_error_rate(predictions, examples):
    label_column = -1
    incorrect_predictions = 0
    
    for i, example in enumerate(examples):
        actual_label = example[label_column]
        predicted_label = predictions[i]
        
        if predicted_label != actual_label:
            incorrect_predictions += 1

    total_samples = len(examples)
    error_rate = incorrect_predictions / total_samples

    return error_rate

def plot_error_rates(train_data, test_data, attributes, attribute_names, attribute_indices, T, feature_sizes):
    for size in feature_sizes:
        train_errors = []
        test_errors = []
        trees = []
        for t in range(1, T + 1):
            print(t)
            new_trees = random_forest(train_data, attributes, attribute_names, T=1, feature_subset_size=size)
            trees.extend(new_trees)  
            train_predictions = predict_bagging(trees, train_data, attribute_indices)
            test_predictions= predict_bagging(trees, test_data, attribute_indices)
            train_error  = calculate_error_rate(train_predictions, train_data)
            test_error = calculate_error_rate(test_predictions, test_data)
            train_errors.append(train_error)
            test_errors.append(test_error)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, T + 1), train_errors, label='Train Error')
        plt.plot(range(1, T + 1), test_errors, label='Test Error')
        plt.title(f'Random Forest Performance with Feature Subset Size: {size}')
        plt.xlabel('Number of Trees')
        plt.ylabel('Error Rate')
        plt.legend()
        plt.grid(True)
        plt.show()

def calculate_bias_variance_error(single_tree_predictions, test_data):
    mean_predictions_single = []
    for predictions in single_tree_predictions:
        mean_prediction = sum(predictions) / len(predictions)
        mean_predictions_single.append(mean_prediction)

    total_bias_squared = 0
    total_variance = 0

    for i in range(len(mean_predictions_single)):
        bias_squared = (mean_predictions_single[i] - test_data[i][-1]) ** 2
        total_bias_squared += bias_squared

        variance = sum((prediction - mean_predictions_single[i]) ** 2 for prediction in single_tree_predictions[i]) / len(single_tree_predictions[i])
        total_variance += variance

    average_bias_squared = total_bias_squared / len(mean_predictions_single)
    average_variance = total_variance / len(mean_predictions_single)
    general_squared_error = average_bias_squared + average_variance

    return average_bias_squared, average_variance, general_squared_error

attribute_names = {
    0: 'age',
    1: 'job',
    2: 'marital',
    3: 'education',
    4: 'default',
    5: 'balance',
    6: 'housing',
    7: 'loan',
    8: 'contact',
    9: 'day',
    10: 'month',
    11: 'duration',
    12: 'campaign',
    13: 'pdays',
    14: 'previous',
    15: 'poutcome'
}

attribute_indices = {
    'age': 0,
    'job': 1,
    'marital': 2,
    'education': 3,
    'default': 4,
    'balance': 5,
    'housing': 6,
    'loan': 7,
    'contact': 8,
    'day': 9,
    'month': 10,
    'duration': 11,
    'campaign': 12,
    'pdays': 13,
    'previous': 14,
    'poutcome': 15
}

train_data = numerical_to_binary('bank/train.csv', attribute_indices)
test_data = numerical_to_binary('bank/test.csv', attribute_indices)
attributes = set(range(16))

all_whole_forest = []
all_single_trees = []

for t in range(100):
    print(t)
    trees = random_forest(train_data, attributes, attribute_names, T=500, feature_subset_size=4)
    all_whole_forest.append(trees)
    all_single_trees.append(trees[0])

single_tree_predictions = []
whole_forest_predictions = []

for example in test_data:
    predictions = [predict(tree, example, attribute_indices) for tree in all_single_trees]
    single_tree_predictions.append(predictions)

whole_forest_predictions = [predict_bagging(model, test_data, attribute_indices) for model in all_whole_forest]

for predictions, model_name in [(single_tree_predictions, "Single Tree"), (whole_forest_predictions, "Whole Forest")]:
    average_bias_squared, average_variance, general_squared_error = calculate_bias_variance_error(predictions, test_data)
    print(f"{model_name}: General Bias: {average_bias_squared}")
    print(f"{model_name}: General Variance: {average_variance}")
    print(f"{model_name}: General Squared Error: {general_squared_error}")