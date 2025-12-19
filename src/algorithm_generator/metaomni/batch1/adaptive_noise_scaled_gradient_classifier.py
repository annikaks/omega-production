import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class AdaptiveNoiseScaledGradientClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that decomposes gradient variance into bias and noise components,
    then adaptively scales learning rate inversely to estimated noise magnitude per parameter.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Base learning rate for gradient descent.
    n_iterations : int, default=1000
        Number of training iterations.
    batch_size : int, default=32
        Size of mini-batches for stochastic gradient descent.
    noise_window : int, default=10
        Window size for estimating gradient noise.
    epsilon : float, default=1e-8
        Small constant for numerical stability.
    alpha : float, default=0.9
        Exponential moving average coefficient for gradient statistics.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32,
                 noise_window=10, epsilon=1e-8, alpha=0.9, random_state=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.noise_window = noise_window
        self.epsilon = epsilon
        self.alpha = alpha
        self.random_state = random_state
    
    def _sigmoid(self, z):
        """Sigmoid activation function with numerical stability."""
        return np.where(z >= 0, 
                       1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))
    
    def _softmax(self, z):
        """Softmax activation function with numerical stability."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _compute_gradient(self, X, y, weights, bias):
        """Compute gradients for weights and bias."""
        n_samples = X.shape[0]
        
        # Forward pass
        logits = np.dot(X, weights) + bias
        
        if self.n_classes_ == 2:
            predictions = self._sigmoid(logits).ravel()
            error = predictions - y
            grad_weights = np.dot(X.T, error) / n_samples
            grad_bias = np.mean(error)
        else:
            predictions = self._softmax(logits)
            y_one_hot = np.zeros((n_samples, self.n_classes_))
            y_one_hot[np.arange(n_samples), y] = 1
            error = predictions - y_one_hot
            grad_weights = np.dot(X.T, error) / n_samples
            grad_bias = np.mean(error, axis=0)
        
        return grad_weights, grad_bias
    
    def _update_noise_estimates(self, grad_weights, grad_bias):
        """Update gradient noise estimates using variance decomposition."""
        # Store gradients in history
        self.grad_history_weights_.append(grad_weights.copy())
        self.grad_history_bias_.append(grad_bias.copy())
        
        # Keep only recent gradients
        if len(self.grad_history_weights_) > self.noise_window:
            self.grad_history_weights_.pop(0)
            self.grad_history_bias_.pop(0)
        
        if len(self.grad_history_weights_) >= 2:
            # Compute mean gradient (bias component)
            mean_grad_weights = np.mean(self.grad_history_weights_, axis=0)
            mean_grad_bias = np.mean(self.grad_history_bias_, axis=0)
            
            # Compute variance (noise component)
            var_grad_weights = np.var(self.grad_history_weights_, axis=0)
            var_grad_bias = np.var(self.grad_history_bias_, axis=0)
            
            # Update noise estimates with exponential moving average
            self.noise_weights_ = (self.alpha * self.noise_weights_ + 
                                  (1 - self.alpha) * np.sqrt(var_grad_weights + self.epsilon))
            self.noise_bias_ = (self.alpha * self.noise_bias_ + 
                               (1 - self.alpha) * np.sqrt(var_grad_bias + self.epsilon))
    
    def _adaptive_learning_rates(self):
        """Compute adaptive learning rates inversely proportional to noise magnitude."""
        # Scale learning rate inversely to noise
        lr_weights = self.learning_rate / (self.noise_weights_ + self.epsilon)
        lr_bias = self.learning_rate / (self.noise_bias_ + self.epsilon)
        
        # Clip learning rates to prevent instability
        lr_weights = np.clip(lr_weights, 0, 10 * self.learning_rate)
        lr_bias = np.clip(lr_bias, 0, 10 * self.learning_rate)
        
        return lr_weights, lr_bias
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier to training data.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Fitted classifier.
        """
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y_train)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        
        # Set random state
        rng = np.random.RandomState(self.random_state)
        
        # Initialize parameters
        n_features = X_train.shape[1]
        
        if self.n_classes_ == 2:
            self.weights_ = rng.randn(n_features) * 0.01
            self.bias_ = 0.0
        else:
            self.weights_ = rng.randn(n_features, self.n_classes_) * 0.01
            self.bias_ = np.zeros(self.n_classes_)
        
        # Initialize noise estimates
        self.noise_weights_ = np.ones_like(self.weights_)
        self.noise_bias_ = np.ones_like(self.bias_)
        
        # Initialize gradient history
        self.grad_history_weights_ = []
        self.grad_history_bias_ = []
        
        # Training loop
        n_samples = X_train.shape[0]
        
        for iteration in range(self.n_iterations):
            # Create mini-batch
            indices = rng.choice(n_samples, size=min(self.batch_size, n_samples), 
                               replace=False)
            X_batch = X_train[indices]
            y_batch = y_encoded[indices]
            
            # Compute gradients
            grad_weights, grad_bias = self._compute_gradient(
                X_batch, y_batch, self.weights_, self.bias_
            )
            
            # Update noise estimates
            self._update_noise_estimates(grad_weights, grad_bias)
            
            # Get adaptive learning rates
            lr_weights, lr_bias = self._adaptive_learning_rates()
            
            # Update parameters with adaptive learning rates
            self.weights_ -= lr_weights * grad_weights
            self.bias_ -= lr_bias * grad_bias
        
        return self
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        probabilities : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        logits = np.dot(X_test, self.weights_) + self.bias_
        
        if self.n_classes_ == 2:
            proba_positive = self._sigmoid(logits).ravel()
            return np.vstack([1 - proba_positive, proba_positive]).T
        else:
            return self._softmax(logits)
    
    def predict(self, X_test):
        """
        Predict class labels for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        predictions : array of shape (n_samples,)
            Predicted class labels.
        """
        probabilities = self.predict_proba(X_test)
        y_pred_encoded = np.argmax(probabilities, axis=1)
        return self.label_encoder_.inverse_transform(y_pred_encoded)