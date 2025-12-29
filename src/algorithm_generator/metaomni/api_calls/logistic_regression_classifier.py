import numpy as np
from sklearn.base import BaseEstimator


class LogisticRegressionClassifier(BaseEstimator):
    def __init__(self, learning_rate=0.01, max_iterations=100):
        """
        Logistic Regression classifier with sigmoid (binary) or softmax (multi-class).
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        max_iterations : int, default=100
            Maximum number of iterations for gradient descent
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
    def fit(self, X_train, y_train):
        """
        Fit the logistic regression model.
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            Training data
        y_train : array-like, shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
        """
        X = np.array(X_train)
        y = np.array(y_train)
        
        # Add bias term
        n_samples, n_features = X.shape
        X_bias = np.c_[np.ones(n_samples), X]
        
        # Determine if binary or multi-class
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Create label mapping
        self.class_to_index_ = {cls: idx for idx, cls in enumerate(self.classes_)}
        
        if n_classes == 2:
            # Binary classification
            self.is_binary_ = True
            # Convert labels to 0 and 1
            y_binary = np.array([self.class_to_index_[label] for label in y])
            
            # Initialize weights
            self.weights_ = np.zeros(n_features + 1)
            
            # Gradient descent
            for _ in range(self.max_iterations):
                # Forward pass - sigmoid
                z = X_bias @ self.weights_
                predictions = self._sigmoid(z)
                
                # Compute gradient
                gradient = X_bias.T @ (predictions - y_binary) / n_samples
                
                # Update weights
                self.weights_ -= self.learning_rate * gradient
                
        else:
            # Multi-class classification
            self.is_binary_ = False
            # Convert labels to indices
            y_indices = np.array([self.class_to_index_[label] for label in y])
            
            # One-hot encode labels
            y_onehot = np.zeros((n_samples, n_classes))
            y_onehot[np.arange(n_samples), y_indices] = 1
            
            # Initialize weights
            self.weights_ = np.zeros((n_features + 1, n_classes))
            
            # Gradient descent
            for _ in range(self.max_iterations):
                # Forward pass - softmax
                z = X_bias @ self.weights_
                predictions = self._softmax(z)
                
                # Compute gradient
                gradient = X_bias.T @ (predictions - y_onehot) / n_samples
                
                # Update weights
                self.weights_ -= self.learning_rate * gradient
        
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
        X = np.array(X_test)
        n_samples = X.shape[0]
        
        # Add bias term
        X_bias = np.c_[np.ones(n_samples), X]
        
        if self.is_binary_:
            # Binary classification
            z = X_bias @ self.weights_
            probabilities = self._sigmoid(z)
            predictions_indices = (probabilities >= 0.5).astype(int)
        else:
            # Multi-class classification
            z = X_bias @ self.weights_
            probabilities = self._softmax(z)
            predictions_indices = np.argmax(probabilities, axis=1)
        
        # Convert indices back to original class labels
        predictions = np.array([self.classes_[idx] for idx in predictions_indices])
        
        return predictions
    
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
        X = np.array(X_test)
        n_samples = X.shape[0]
        
        # Add bias term
        X_bias = np.c_[np.ones(n_samples), X]
        
        if self.is_binary_:
            # Binary classification
            z = X_bias @ self.weights_
            prob_class_1 = self._sigmoid(z)
            probabilities = np.c_[1 - prob_class_1, prob_class_1]
        else:
            # Multi-class classification
            z = X_bias @ self.weights_
            probabilities = self._softmax(z)
        
        return probabilities
    
    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _softmax(self, z):
        """Softmax activation function."""
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)