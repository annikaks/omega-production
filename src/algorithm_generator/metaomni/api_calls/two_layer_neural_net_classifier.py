import numpy as np
from sklearn.base import BaseEstimator


class TwoLayerNeuralNetClassifier(BaseEstimator):
    def __init__(self, hidden_size=64, learning_rate=0.01, n_epochs=1000, 
                 batch_size=32, random_state=None, verbose=False):
        """
        Two-layer fully connected neural network classifier.
        
        Parameters
        ----------
        hidden_size : int, default=64
            Number of neurons in the hidden layer
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        n_epochs : int, default=1000
            Number of training epochs
        batch_size : int, default=32
            Size of mini-batches for training
        random_state : int, default=None
            Random seed for reproducibility
        verbose : bool, default=False
            Whether to print training progress
        """
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        
    def _initialize_weights(self, n_features, n_classes):
        """Initialize weights and biases."""
        rng = np.random.RandomState(self.random_state)
        
        # Xavier initialization
        self.W1 = rng.randn(n_features, self.hidden_size) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = rng.randn(self.hidden_size, n_classes) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros((1, n_classes))
        
    def _relu(self, Z):
        """ReLU activation function."""
        return np.maximum(0, Z)
    
    def _relu_derivative(self, Z):
        """Derivative of ReLU."""
        return (Z > 0).astype(float)
    
    def _softmax(self, Z):
        """Softmax activation function."""
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def _forward_pass(self, X):
        """Forward propagation."""
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self._relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self._softmax(self.Z2)
        return self.A2
    
    def _backward_pass(self, X, y_one_hot):
        """Backward propagation."""
        m = X.shape[0]
        
        # Output layer gradients
        dZ2 = self.A2 - y_one_hot
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self._relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        return dW1, db1, dW2, db2
    
    def _update_weights(self, dW1, db1, dW2, db2):
        """Update weights using gradient descent."""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def _compute_loss(self, y_pred, y_one_hot):
        """Compute cross-entropy loss."""
        m = y_one_hot.shape[0]
        log_probs = -np.log(y_pred[range(m), np.argmax(y_one_hot, axis=1)] + 1e-8)
        return np.mean(log_probs)
    
    def fit(self, X_train, y_train):
        """
        Fit the neural network classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Returns self.
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Store classes
        self.classes_ = np.unique(y_train)
        n_classes = len(self.classes_)
        n_samples, n_features = X_train.shape
        
        # Create class mapping
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        
        # Convert labels to one-hot encoding
        y_indices = np.array([self.class_to_idx_[y] for y in y_train])
        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y_indices] = 1
        
        # Initialize weights
        self._initialize_weights(n_features, n_classes)
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_one_hot[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]
                
                # Forward pass
                self._forward_pass(X_batch)
                
                # Backward pass
                dW1, db1, dW2, db2 = self._backward_pass(X_batch, y_batch)
                
                # Update weights
                self._update_weights(dW1, db1, dW2, db2)
            
            # Print progress
            if self.verbose and (epoch + 1) % 100 == 0:
                y_pred = self._forward_pass(X_train)
                loss = self._compute_loss(y_pred, y_one_hot)
                accuracy = np.mean(np.argmax(y_pred, axis=1) == y_indices)
                print(f"Epoch {epoch + 1}/{self.n_epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        X_test = np.array(X_test)
        y_pred_proba = self._forward_pass(X_test)
        y_pred_indices = np.argmax(y_pred_proba, axis=1)
        return np.array([self.classes_[i] for i in y_pred_indices])
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        y_pred_proba : array of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        X_test = np.array(X_test)
        return self._forward_pass(X_test)