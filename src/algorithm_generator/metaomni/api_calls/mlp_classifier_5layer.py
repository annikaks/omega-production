import numpy as np
from sklearn.base import BaseEstimator


class MLPClassifier5Layer(BaseEstimator):
    def __init__(self, hidden_units=(128, 64, 32, 16), learning_rate=0.01, 
                 epochs=100, batch_size=32, random_state=None):
        """
        5-layer fully connected neural network classifier.
        
        Parameters:
        -----------
        hidden_units : tuple of 4 ints
            Number of units in each of the 4 hidden layers
        learning_rate : float
            Learning rate for gradient descent
        epochs : int
            Number of training epochs
        batch_size : int
            Mini-batch size for training
        random_state : int or None
            Random seed for reproducibility
        """
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _initialize_weights(self, n_features, n_classes):
        """Initialize weights and biases for all 5 layers."""
        np.random.seed(self.random_state)
        
        layer_sizes = [n_features] + list(self.hidden_units) + [n_classes]
        self.weights_ = []
        self.biases_ = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights_.append(w)
            self.biases_.append(b)
    
    def _forward_pass(self, X):
        """Forward propagation through all layers."""
        self.activations_ = [X]
        
        # Hidden layers (4 layers with sigmoid)
        for i in range(4):
            z = np.dot(self.activations_[-1], self.weights_[i]) + self.biases_[i]
            a = self._sigmoid(z)
            self.activations_.append(a)
        
        # Output layer (softmax)
        z = np.dot(self.activations_[-1], self.weights_[4]) + self.biases_[4]
        a = self._softmax(z)
        self.activations_.append(a)
        
        return self.activations_[-1]
    
    def _backward_pass(self, X, y):
        """Backward propagation through all layers."""
        m = X.shape[0]
        
        # Convert y to one-hot encoding
        y_one_hot = np.zeros((m, self.n_classes_))
        y_one_hot[np.arange(m), y] = 1
        
        # Output layer gradient
        delta = self.activations_[-1] - y_one_hot
        
        gradients_w = []
        gradients_b = []
        
        # Backpropagate through all layers
        for i in range(4, -1, -1):
            grad_w = np.dot(self.activations_[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)
            
            if i > 0:
                delta = np.dot(delta, self.weights_[i].T) * self._sigmoid_derivative(self.activations_[i])
        
        return gradients_w, gradients_b
    
    def fit(self, X_train, y_train):
        """
        Fit the neural network classifier.
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            Training data
        y_train : array-like, shape (n_samples,)
            Target labels
        
        Returns:
        --------
        self : object
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.classes_ = np.unique(y_train)
        self.n_classes_ = len(self.classes_)
        
        # Map labels to indices
        self.label_map_ = {label: idx for idx, label in enumerate(self.classes_)}
        y_mapped = np.array([self.label_map_[label] for label in y_train])
        
        n_samples, n_features = X_train.shape
        self._initialize_weights(n_features, self.n_classes_)
        
        # Training loop
        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_mapped[indices]
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Forward pass
                self._forward_pass(X_batch)
                
                # Backward pass
                gradients_w, gradients_b = self._backward_pass(X_batch, y_batch)
                
                # Update weights
                for j in range(5):
                    self.weights_[j] -= self.learning_rate * gradients_w[j]
                    self.biases_[j] -= self.learning_rate * gradients_b[j]
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test data
        
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        X_test = np.array(X_test)
        probabilities = self._forward_pass(X_test)
        predicted_indices = np.argmax(probabilities, axis=1)
        
        # Map indices back to original labels
        return np.array([self.classes_[idx] for idx in predicted_indices])
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test data
        
        Returns:
        --------
        probabilities : array, shape (n_samples, n_classes)
            Predicted class probabilities
        """
        X_test = np.array(X_test)
        return self._forward_pass(X_test)