import csv
import math
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

def weighted_error_rate(predictions, actuals, weights):
    return np.sum(weights * (predictions != actuals)) / np.sum(weights)

def draw(train_errors, test_errors, train_stump_errors, test_stump_errors, T):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot for AdaBoost errors
    ax1.plot(range(1, T+1), train_errors, label='Training Error')
    ax1.plot(range(1, T+1), test_errors, label='Testing Error')
    ax1.set_xlabel('Number of Iterations (T)')
    ax1.set_ylabel('Error Rate')
    ax1.set_title('AdaBoost Training and Testing Errors')
    ax1.legend()
    ax1.grid(True)
    
    # Plot for decision stump errors
    ax2.plot(range(1, T+1), train_stump_errors, label='Stump Training Error')
    ax2.plot(range(1, T+1), test_stump_errors, label='Stump Testing Error')
    ax2.set_xlabel('Number of Iterations (T)')
    ax2.set_ylabel('Error Rate')
    ax2.set_title('Decision Stump Training and Testing Errors')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def adaboost(train_data, test_data, attributes, attribute_names, T, attribute_indices):
    n = len(train_data)
    weights = np.ones(n) / n
    max_depth = 1
    train_errors = []
    test_errors = []
    train_stump_errors = []
    test_stump_errors = []
    for t in range(T):
        print(t)
        stump = id3(train_data, attributes, attribute_names, 'entropy', max_depth, weights)

        train_predictions = np.array([predict(stump, example, attribute_indices) for example in train_data])
        train_actuals = np.array([example[-1] for example in train_data])
        train_error = weighted_error_rate(train_predictions, train_actuals, weights)
        
        test_predictions = np.array([predict(stump, example, attribute_indices) for example in test_data])
        test_actuals = np.array([example[-1] for example in test_data])
        test_error = weighted_error_rate(test_predictions, test_actuals, weights)


        alpha = 0.5 * np.log((1 - train_error) / train_error)

        weights *= np.exp(-alpha * train_actuals  * train_predictions)
        weights /= np.sum(weights)

        train_errors.append(train_error)
        test_errors.append(test_error)
        train_stump_errors.append(calculate_error_rate(train_data, stump, attribute_indices))
        test_stump_errors.append(calculate_error_rate(test_data, stump, attribute_indices))

    return train_errors, test_errors, train_stump_errors, test_stump_errors

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

def calculate_entropy(examples, label_column, weights):
    weighted_label_counts = {}

    for i, example in enumerate(examples):
        label = example[label_column]
        weight = weights[i]
        if label in weighted_label_counts:
            weighted_label_counts[label] += weight
        else:
            weighted_label_counts[label] = weight

    total_weight = sum(weighted_label_counts.values())

    if total_weight == 0:
        return 0

    entropy = 0
    for count in weighted_label_counts.values():
        probability = count / total_weight
        entropy -= probability * math.log2(probability)

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

def calculate_measure(examples, method, label_column, weights):
    if method == 'entropy':
        return calculate_entropy(examples, label_column, weights)
    elif method == 'majority_error':
        return calculate_majority_error(examples, label_column)
    elif method == 'gini':
        return calculate_gini(examples, label_column)

    raise 'The method is invalid'

def best_attribute(examples, attributes, method, label_column, weights):
    best_gain = -1
    best_attribute = None

    for attribute in attributes:
        gain = calculate_information_gain(examples, attribute, method, label_column, weights)
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute

    return best_attribute

def calculate_information_gain(examples, attribute, method, label_column, weights):
    base_measure = calculate_measure(examples, method, label_column, weights)

    attribute_values = set(example[attribute] for example in examples)
    weighted_sum = 0

    for value in attribute_values:
        subset = [example for example in examples if example[attribute] == value]
        subset_weights = [weights[i] for i, example in enumerate(examples) if example[attribute] == value]
        subset_measure = calculate_measure(subset, method, label_column, subset_weights)
        weighted_sum += sum(subset_weights) * subset_measure

    information_gain = base_measure - weighted_sum / sum(weights)
    return information_gain

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

def id3(examples, attributes, attribute_names, method, max_depth, weights, current_depth=0):
    label_column = -1
    
    labels = [example[label_column] for example in examples]
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    weighted_counts = label_counts * weights[:len(unique_labels)]
    majority_label = unique_labels[np.argmax(weighted_counts)]

    if len(unique_labels) == 1 or current_depth == max_depth:
        return majority_label

    best_attribute_index = best_attribute(examples, attributes, method, label_column, weights)
    best_attribute_name = attribute_names[best_attribute_index]

    tree = {best_attribute_name: {}}

    for value in get_all_possible_values_for_attribute(best_attribute_name):
        subset = [example for example in examples if example[best_attribute_index] == value]
        subset_weights = [weights[i] for i, example in enumerate(examples) if example[best_attribute_index] == value]
        
        if not subset:
            tree[best_attribute_name][value] = majority_label
        else:
            remaining_attributes = attributes - {best_attribute_index}
            subtree = id3(subset, remaining_attributes, attribute_names, method, max_depth, subset_weights, current_depth + 1)
            tree[best_attribute_name][value] = subtree

    return tree

def predict(tree, example, attribute_indices):
    if not isinstance(tree, dict):
        return tree
    
    attribute = next(iter(tree))
    attribute_value = example[attribute_indices[attribute]]
    
    if attribute_value in tree[attribute]:
        return predict(tree[attribute][attribute_value], example, attribute_indices)
    else:
        return max(set(tree[attribute].values()), key=list(tree[attribute].values()).count)

def calculate_error_rate(data, tree, attribute_indices):
    label_column = -1
    incorrect_predictions = 0
    for example in data:
        actual_label = example[label_column]
        predicted_label = predict(tree, example, attribute_indices)
        if predicted_label != actual_label:
            incorrect_predictions += 1

    total = len(data)
    error_rate = incorrect_predictions / total
    return error_rate

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
T=500
train_errors, test_errors, train_stump_errors, test_stump_errors = adaboost(train_data, test_data, attributes, attribute_names, T, attribute_indices)
draw(train_errors, test_errors, train_stump_errors, test_stump_errors, T)