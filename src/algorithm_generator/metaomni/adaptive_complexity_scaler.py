import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

class DynamicNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(DynamicNet, self).__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def add_layer(self, size):
        new_layer = nn.Linear(size, size)
        new_activation = nn.ReLU()
        self.layers.insert(-1, new_layer)
        self.layers.insert(-1, new_activation)

class AdaptiveComplexityNet(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_size=64, max_epochs=100, batch_size=32, learning_rate=0.001,
                 complexity_threshold=0.95, patience=5):
        self.hidden_size = hidden_size
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.complexity_threshold = complexity_threshold
        self.patience = patience

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
        
        # Initialize the model
        self.model_ = DynamicNet(n_features, n_classes, self.hidden_size)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        best_val_acc = 0
        epochs_without_improvement = 0
        
        for epoch in range(self.max_epochs):
            # Training
            self.model_.train()
            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i+self.batch_size]
                batch_y = y_train[i:i+self.batch_size]
                
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model_.eval()
            with torch.no_grad():
                val_outputs = self.model_(X_val)
                val_preds = torch.argmax(val_outputs, dim=1)
                val_acc = accuracy_score(y_val, val_preds)
            
            # Check if we need to increase complexity
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if val_acc < self.complexity_threshold and epochs_without_improvement >= self.patience:
                self.model_.add_layer(self.hidden_size)
                optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
                epochs_without_improvement = 0
            
            # Early stopping
            if epochs_without_improvement >= self.patience * 2:
                break
        
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        X_tensor = torch.FloatTensor(X)
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        
        return self.classes_[predictions]