import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class FractalUnit:
    def __init__(self, input_dim, output_dim, depth):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.zeros(output_dim)
        
        if depth > 0:
            self.sub_units = [
                FractalUnit(output_dim, output_dim, depth - 1)
                for _ in range(2)
            ]
        else:
            self.sub_units = None
    
    def forward(self, X):
        output = np.dot(X, self.weights) + self.bias
        output = np.maximum(output, 0)  # ReLU activation
        
        if self.sub_units:
            sub_output1 = self.sub_units[0].forward(output)
            sub_output2 = self.sub_units[1].forward(output)
            output = (sub_output1 + sub_output2) / 2
        
        return output
    
    def backward(self, X, grad_output, learning_rate):
        if self.sub_units:
            grad_sub = grad_output / 2
            grad_output = (
                self.sub_units[0].backward(X, grad_sub, learning_rate) +
                self.sub_units[1].backward(X, grad_sub, learning_rate)
            ) / 2
        
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(X.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0)
        
        self.weights -= learning_rate * grad_weights
        self.bias -= learning_rate * grad_bias
        
        return grad_input

class FractalNetArch(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dim, output_dim, fractal_depth=3, learning_rate=0.01, epochs=100):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fractal_depth = fractal_depth
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        
        self.fractal_unit = FractalUnit(self.input_dim, self.hidden_dim, self.fractal_depth)
        self.output_layer = np.random.randn(self.hidden_dim, self.output_dim)
        self.output_bias = np.zeros(self.output_dim)
        
        for _ in range(self.epochs):
            # Forward pass
            hidden = self.fractal_unit.forward(X)
            output = np.dot(hidden, self.output_layer) + self.output_bias
            probabilities = self._softmax(output)
            
            # Backward pass
            y_one_hot = np.eye(self.output_dim)[y]
            grad_output = probabilities - y_one_hot
            grad_hidden = np.dot(grad_output, self.output_layer.T)
            
            self.fractal_unit.backward(X, grad_hidden, self.learning_rate)
            
            self.output_layer -= self.learning_rate * np.dot(hidden.T, grad_output)
            self.output_bias -= self.learning_rate * np.sum(grad_output, axis=0)
        
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        hidden = self.fractal_unit.forward(X)
        output = np.dot(hidden, self.output_layer) + self.output_bias
        return self.classes_[np.argmax(output, axis=1)]
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)