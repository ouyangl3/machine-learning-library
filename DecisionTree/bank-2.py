import csv
import math
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

def convert_missing_values(data, attribute_indices):
    numerical_features = ['job', 'education', 'contact', 'poutcome']

    for feature in numerical_features:
        column_name = data.columns[attribute_indices[feature]]  # Get the column name

        # Calculate the most common value for the column excluding 'unknown'
        majority_value = data[column_name][data[column_name] != 'unknown'].mode()[0]

        # Replace 'unknown' with the majority value
        data[column_name] = np.where(data[column_name] == 'unknown', majority_value, data[column_name])

    return data


def numerical_to_binary(filepath, attribute_indices):
    data = pd.read_csv(filepath, header=None)
    numerical_features = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    for feature in numerical_features:
        column_name = data.columns[attribute_indices[feature]]
        median_value = data[column_name].median()

        # 1 if above median, 0 otherwise
        data[column_name] = np.where(data[column_name].astype(float) > median_value, 1, 0)

    data = convert_missing_values(data, attribute_indices)
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

def id3(examples, attributes, attribute_names, method, max_depth, current_depth=0):
    label_column = -1
    # If all examples have same label, return a leaf node with the label
    if len(set(example[label_column] for example in examples)) == 1:
        return examples[0][label_column]

    # If Attributes is empty or if the maximum depth has been reached, return a leaf node with the majority label
    if not attributes or current_depth == max_depth:
        return majority_label(examples, label_column)

    best_attribute_index = best_attribute(examples, attributes, method, label_column)
    best_attribute_name = attribute_names[best_attribute_index]

    tree = {best_attribute_name: {}}

    all_possible_values = get_all_possible_values_for_attribute(best_attribute_name)

    for value in all_possible_values:
        subset = [example for example in examples if example[best_attribute_index] == value]

        # For each value, recurse to create subtrees
        if subset:
            # Remove the best attribute for further splits
            attributes = attributes - {best_attribute_index}
            tree[best_attribute_name][value] = id3(subset, attributes, attribute_names, method, max_depth, current_depth + 1)
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

methods = ['entropy', 'majority_error', 'gini']
for method in methods:
    print(method)
    for max_depth in range(1,17):
        decision_tree = id3(train_data, attributes, attribute_names, method, max_depth)
        # print(decision_tree)
        error = calculate_error_rate(test_data, decision_tree, attribute_indices)
        print(f'{max_depth}. The Average Prediction Error Rate: {error:.4f}')
