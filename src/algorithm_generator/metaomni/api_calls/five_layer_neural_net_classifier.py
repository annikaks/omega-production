import numpy as np
from sklearn.base import BaseEstimator


class FiveLayerNeuralNetClassifier(BaseEstimator):
    def __init__(self, hidden_sizes=(128, 64, 32, 16), learning_rate=0.01, 
                 n_epochs=100, batch_size=32, random_state=None, 
                 activation='relu', verbose=False):
        """
        5-layer fully connected neural network classifier.
        
        Parameters:
        -----------
        hidden_sizes : tuple of 4 ints
            Sizes of the 4 hidden layers (5 layers total including output)
        learning_rate : float
            Learning rate for gradient descent
        n_epochs : int
            Number of training epochs
        batch_size : int
            Mini-batch size for training
        random_state : int or None
            Random seed for reproducibility
        activation : str
            Activation function ('relu', 'tanh', or 'sigmoid')
        verbose : bool
            Whether to print training progress
        """
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.activation = activation
        self.verbose = verbose
        
    def _initialize_weights(self, n_features, n_classes):
        """Initialize weights and biases for all layers."""
        np.random.seed(self.random_state)
        
        layer_sizes = [n_features] + list(self.hidden_sizes) + [n_classes]
        self.weights_ = []
        self.biases_ = []
        
        for i in range(len(layer_sizes) - 1):
            # He initialization for better convergence
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights_.append(w)
            self.biases_.append(b)
    
    def _activate(self, z):
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _activate_derivative(self, z):
        """Compute derivative of activation function."""
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation == 'sigmoid':
            s = self._activate(z)
            return s * (1 - s)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _softmax(self, z):
        """Compute softmax for output layer."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _forward_pass(self, X):
        """Forward propagation through the network."""
        activations = [X]
        z_values = []
        
        # Hidden layers
        for i in range(len(self.weights_) - 1):
            z = activations[-1] @ self.weights_[i] + self.biases_[i]
            z_values.append(z)
            a = self._activate(z)
            activations.append(a)
        
        # Output layer
        z = activations[-1] @ self.weights_[-1] + self.biases_[-1]
        z_values.append(z)
        output = self._softmax(z)
        activations.append(output)
        
        return activations, z_values
    
    def _backward_pass(self, X, y, activations, z_values):
        """Backward propagation to compute gradients."""
        m = X.shape[0]
        n_layers = len(self.weights_)
        
        # Convert y to one-hot encoding
        y_one_hot = np.zeros((m, self.n_classes_))
        y_one_hot[np.arange(m), y] = 1
        
        # Initialize gradients
        dW = [None] * n_layers
        db = [None] * n_layers
        
        # Output layer gradient
        delta = activations[-1] - y_one_hot
        dW[-1] = activations[-2].T @ delta / m
        db[-1] = np.sum(delta, axis=0, keepdims=True) / m
        
        # Hidden layers gradients
        for i in range(n_layers - 2, -1, -1):
            delta = (delta @ self.weights_[i+1].T) * self._activate_derivative(z_values[i])
            dW[i] = activations[i].T @ delta / m
            db[i] = np.sum(delta, axis=0, keepdims=True) / m
        
        return dW, db
    
    def _update_weights(self, dW, db):
        """Update weights and biases using gradient descent."""
        for i in range(len(self.weights_)):
            self.weights_[i] -= self.learning_rate * dW[i]
            self.biases_[i] -= self.learning_rate * db[i]
    
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
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_mapped[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Forward pass
                activations, z_values = self._forward_pass(X_batch)
                
                # Backward pass
                dW, db = self._backward_pass(X_batch, y_batch, activations, z_values)
                
                # Update weights
                self._update_weights(dW, db)
            
            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                activations, _ = self._forward_pass(X_train)
                predictions = np.argmax(activations[-1], axis=1)
                accuracy = np.mean(predictions == y_mapped)
                print(f"Epoch {epoch+1}/{self.n_epochs}, Accuracy: {accuracy:.4f}")
        
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
        activations, _ = self._forward_pass(X_test)
        predictions = np.argmax(activations[-1], axis=1)
        
        # Map back to original labels
        return np.array([self.classes_[idx] for idx in predictions])
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test data
        
        Returns:
        --------
        proba : array, shape (n_samples, n_classes)
            Predicted class probabilities
        """
        X_test = np.array(X_test)
        activations, _ = self._forward_pass(X_test)
        return activations[-1]