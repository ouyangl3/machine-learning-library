# DecisionTree
1. Load Data: The function load_data(filepath) is used to load data from a CSV file into a list.

2. Set Attributes: The attributes for splitting the data are represented as a set of indices (e.g., 0 for 'buying', 1 for 'maint', etc.). 
    Initialize the attributes for the tree as:
    attributes = set(range(6))  # Assumes 6 attributes for the car dataset

3. Attribute Names: A dictionary maps the indices of attributes to their names:
    attribute_names = {
        0: 'buying',
        1: 'maint',
        2: 'doors',
        3: 'persons',
        4: 'lug_boot',
        5: 'safety'
    }

4. Choose a Method: Three methods are supported to evaluate splits:
    Entropy: 'entropy'
    Majority Error: 'majority_error'
    Gini Index: 'gini'

5. Build the Decision Tree: Call the id3 function to build the decision tree. Parameters:
    examples: The training data.
    attributes: A set of attribute indices.
    attribute_names: Dictionary of attribute names.
    method: The splitting method to use.
    max_depth: The maximum depth of the tree.

6. Evaluate the Model: Use the calculate_error_rate function to calculate the error on training or test data:



