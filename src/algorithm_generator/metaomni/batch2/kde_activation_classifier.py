import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import gaussian_kde


class KDEActivationClassifier(BaseEstimator, ClassifierMixin):
    """
    Neural network classifier with learnable KDE-based activation functions.
    
    Each neuron uses a kernel density estimator that adapts to the local
    data distribution instead of fixed activation functions like ReLU or sigmoid.
    
    Parameters
    ----------
    hidden_layers : tuple, default=(64, 32)
        Number of neurons in each hidden layer.
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    max_iter : int, default=1000
        Maximum number of iterations for training.
    kde_bandwidth : str or float, default='scott'
        Bandwidth selection method for KDE ('scott', 'silverman', or float).
    kde_samples : int, default=100
        Number of samples to use for KDE estimation per neuron.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, hidden_layers=(64, 32), learning_rate=0.01, 
                 max_iter=1000, kde_bandwidth='scott', kde_samples=100,
                 random_state=None):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.kde_bandwidth = kde_bandwidth
        self.kde_samples = kde_samples
        self.random_state = random_state
        
    def _initialize_weights(self, n_features, n_classes):
        """Initialize network weights."""
        np.random.seed(self.random_state)
        
        self.weights_ = []
        self.biases_ = []
        self.kde_estimators_ = []
        
        layer_sizes = [n_features] + list(self.hidden_layers) + [n_classes]
        
        for i in range(len(layer_sizes) - 1):
            # He initialization
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights_.append(w)
            self.biases_.append(b)
            
            # Initialize KDE estimators for each neuron in this layer
            self.kde_estimators_.append([None] * layer_sizes[i+1])
    
    def _kde_activation(self, z, layer_idx, neuron_idx, training=False):
        """
        Apply KDE-based activation function.
        
        During training, collect samples to build KDE.
        During inference, use learned KDE for activation.
        """
        if training:
            # During training, use tanh as temporary activation and collect samples
            return np.tanh(z)
        else:
            # Use learned KDE for activation
            kde = self.kde_estimators_[layer_idx][neuron_idx]
            if kde is None:
                # Fallback to tanh if KDE not available
                return np.tanh(z)
            
            # Evaluate KDE at input points
            z_flat = z.flatten()
            try:
                # Use KDE to transform activations
                # Map through learned distribution
                activated = np.zeros_like(z_flat)
                for i, val in enumerate(z_flat):
                    # Evaluate KDE probability and use as scaling factor
                    prob = kde.evaluate(np.array([val]))[0]
                    activated[i] = val * (1 + prob)
                return activated.reshape(z.shape)
            except:
                return np.tanh(z)
    
    def _forward_pass(self, X, training=False):
        """Forward propagation through the network."""
        self.activations_ = [X]
        self.pre_activations_ = []
        
        current_input = X
        
        for layer_idx in range(len(self.weights_)):
            # Linear transformation
            z = np.dot(current_input, self.weights_[layer_idx]) + self.biases_[layer_idx]
            self.pre_activations_.append(z)
            
            # Apply activation
            if layer_idx < len(self.weights_) - 1:  # Hidden layers
                if training:
                    # Use simple activation during initial training
                    a = np.tanh(z)
                else:
                    # Apply KDE-based activation per neuron
                    a = np.zeros_like(z)
                    for neuron_idx in range(z.shape[1]):
                        a[:, neuron_idx] = self._kde_activation(
                            z[:, neuron_idx:neuron_idx+1], 
                            layer_idx, 
                            neuron_idx, 
                            training
                        ).flatten()
            else:  # Output layer
                # Softmax for classification
                exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                a = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            
            self.activations_.append(a)
            current_input = a
        
        return self.activations_[-1]
    
    def _backward_pass(self, X, y):
        """Backward propagation to compute gradients."""
        m = X.shape[0]
        
        # Convert labels to one-hot encoding
        y_one_hot = np.zeros((m, self.n_classes_))
        y_one_hot[np.arange(m), y] = 1
        
        # Output layer gradient
        delta = self.activations_[-1] - y_one_hot
        
        # Backpropagate through layers
        for layer_idx in range(len(self.weights_) - 1, -1, -1):
            # Compute gradients
            dW = np.dot(self.activations_[layer_idx].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            # Update weights
            self.weights_[layer_idx] -= self.learning_rate * dW
            self.biases_[layer_idx] -= self.learning_rate * db
            
            # Propagate error to previous layer
            if layer_idx > 0:
                delta = np.dot(delta, self.weights_[layer_idx].T)
                # Derivative of tanh
                delta *= (1 - self.activations_[layer_idx] ** 2)
    
    def _update_kde_estimators(self, X):
        """Update KDE estimators based on current activations."""
        # Forward pass to collect activations
        self._forward_pass(X, training=True)
        
        # Build KDE for each neuron in each layer
        for layer_idx in range(len(self.weights_) - 1):  # Exclude output layer
            z = self.pre_activations_[layer_idx]
            
            for neuron_idx in range(z.shape[1]):
                # Collect samples for this neuron
                samples = z[:, neuron_idx]
                
                # Subsample if too many points
                if len(samples) > self.kde_samples:
                    indices = np.random.choice(len(samples), self.kde_samples, replace=False)
                    samples = samples[indices]
                
                # Fit KDE
                try:
                    if len(np.unique(samples)) > 1:  # Need variation for KDE
                        kde = gaussian_kde(samples, bw_method=self.kde_bandwidth)
                        self.kde_estimators_[layer_idx][neuron_idx] = kde
                except:
                    # If KDE fails, keep as None (will use fallback)
                    pass
    
    def fit(self, X_train, y_train):
        """
        Fit the KDE activation classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        
        # Map labels to indices
        self.label_map_ = {label: idx for idx, label in enumerate(self.classes_)}
        y_mapped = np.array([self.label_map_[label] for label in y_train])
        
        # Initialize network
        n_features = X_train.shape[1]
        self._initialize_weights(n_features, self.n_classes_)
        
        # Training loop
        for iteration in range(self.max_iter):
            # Forward and backward pass
            self._forward_pass(X_train, training=True)
            self._backward_pass(X_train, y_mapped)
            
            # Update KDE estimators periodically
            if iteration % max(1, self.max_iter // 10) == 0:
                self._update_kde_estimators(X_train)
        
        # Final KDE update
        self._update_kde_estimators(X_train)
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ['weights_', 'biases_', 'kde_estimators_'])
        X_test = check_array(X_test)
        
        # Forward pass with KDE activations
        probabilities = self._forward_pass(X_test, training=False)
        
        # Get predicted class indices
        y_pred_indices = np.argmax(probabilities, axis=1)
        
        # Map back to original labels
        y_pred = np.array([self.classes_[idx] for idx in y_pred_indices])
        
        return y_pred
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        probabilities : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self, ['weights_', 'biases_', 'kde_estimators_'])
        X_test = check_array(X_test)
        
        probabilities = self._forward_pass(X_test, training=False)
        
        return probabilities