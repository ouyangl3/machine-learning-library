import csv
import math
from collections import Counter, defaultdict

# Load data from a csv file
def load_data(filepath):
    with open(filepath, newline='') as f:
        reader = csv.reader(f)
        return list(reader)

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
        'buying':   ['vhigh', 'high', 'med', 'low'],
        'maint':    ['vhigh', 'high', 'med', 'low'],
        'doors':    ['2', '3', '4', '5more'],
        'persons':   ['2', '4', 'more'],
        'lug_boot': ['small', 'med', 'big'],
        'safety':   ['low', 'med', 'high']
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

def predict(tree, example):
    # Base case: if the current node is a leaf node, return its value
    if not isinstance(tree, dict):
        return tree

    attribute_indices = {
        'buying': 0,
        'maint': 1,
        'doors': 2,
        'persons': 3,
        'lug_boot': 4,
        'safety': 5
    }

    # Get the first key in the dictionary as the attribute
    attribute = next(iter(tree))
    attribute_value = example[attribute_indices[attribute]]
    
    # Check if the attribute value has a corresponding subtree
    if attribute_value in tree[attribute]:
        subtree = tree[attribute][attribute_value]
    
    return predict(subtree, example)

def calculate_error_rate(data, tree):
    label_column = -1
    incorrect_predictions = 0
    for example in data:
        actual_label = example[label_column]
        predicted_label = predict(tree, example)
        if predicted_label != actual_label:
            incorrect_predictions += 1

    total = len(data)
    error_rate = incorrect_predictions / total
    return error_rate

train_data = load_data('car/train.csv')
test_data = load_data('car/test.csv')
attributes = set(range(6))

attribute_names = {
    0: 'buying',
    1: 'maint',
    2: 'doors',
    3: 'persons',
    4: 'lug_boot',
    5: 'safety'
}

methods = ['entropy', 'majority_error', 'gini']
for method in methods:
    print(method)
    for max_depth in range(1,7):
        decision_tree = id3(train_data, attributes, attribute_names, method, max_depth)
        # print(decision_tree)
        error = calculate_error_rate(train_data, decision_tree)
        print(f'{max_depth}. The Average Prediction Error Rate: {error:.4f}')