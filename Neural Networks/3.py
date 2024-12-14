import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, width, depth, activation, init):
        super(NeuralNetwork, self).__init__()
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(nn.Linear(input_dim, width))
            else:
                layers.append(nn.Linear(width, width))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 1))  # Output layer
        self.network = nn.Sequential(*layers)
        
        # Apply initialization
        self.apply(init)

    def forward(self, x):
        return self.network(x)

# Xavier Initialization
def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# He Initialization
def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

def train_model(X_train, y_train, X_test, y_test, input_dim, width, depth, activation, init, epochs=50):
    model = NeuralNetwork(input_dim, width, depth, activation, init)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_errors = []
    test_errors = []
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            train_preds = (torch.sigmoid(model(X_train)) > 0.5).float()
            test_preds = (torch.sigmoid(model(X_test)) > 0.5).float()
            train_error = 1 - accuracy_score(y_train, train_preds)
            test_error = 1 - accuracy_score(y_test, test_preds)
            train_errors.append(train_error)
            test_errors.append(test_error)
    
    return train_errors[-1], test_errors[-1]

# Load and preprocess
X, y = load_data("bank-note/train.csv")
X_test, y_test = load_data("bank-note/test.csv")

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

input_dim = X_train.shape[1]
results = []

depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]
activations = ['tanh', 'relu']
inits = {'tanh': xavier_init, 'relu': he_init}

for depth in depths:
    for width in widths:
        for activation in activations:
            init = inits[activation]
            train_err, test_err = train_model(X_train, y_train, X_test, y_test, input_dim, width, depth, activation, init)
            results.append((depth, width, activation, train_err, test_err))
            print(f"Depth={depth}, Width={width}, Activation={activation}: Train Error={train_err:.4f}, Test Error={test_err:.4f}")