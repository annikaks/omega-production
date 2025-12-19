import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class AdaptivePruningClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that dynamically simplifies decision boundaries by pruning
    weight dimensions with consistently near-zero gradients during training.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    
    max_iter : int, default=1000
        Maximum number of training iterations.
    
    gradient_threshold : float, default=1e-4
        Threshold below which gradients are considered near-zero.
    
    prune_patience : int, default=50
        Number of consecutive iterations a gradient must be near-zero before pruning.
    
    complexity_penalty : float, default=0.01
        L2 regularization strength for complexity penalty.
    
    prune_check_interval : int, default=10
        Interval (in iterations) to check for pruning candidates.
    
    tol : float, default=1e-4
        Tolerance for convergence.
    
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, learning_rate=0.01, max_iter=1000, gradient_threshold=1e-4,
                 prune_patience=50, complexity_penalty=0.01, prune_check_interval=10,
                 tol=1e-4, random_state=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.gradient_threshold = gradient_threshold
        self.prune_patience = prune_patience
        self.complexity_penalty = complexity_penalty
        self.prune_check_interval = prune_check_interval
        self.tol = tol
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
    
    def _initialize_weights(self, n_features, n_classes):
        """Initialize weights and bias."""
        rng = np.random.RandomState(self.random_state)
        self.weights_ = rng.randn(n_features, n_classes) * 0.01
        self.bias_ = np.zeros((1, n_classes))
        self.active_features_ = np.ones(n_features, dtype=bool)
        self.gradient_history_ = np.zeros(n_features)
        self.low_gradient_counter_ = np.zeros(n_features)
    
    def _compute_loss(self, X, y_encoded, weights, bias):
        """Compute cross-entropy loss with L2 regularization."""
        n_samples = X.shape[0]
        logits = np.dot(X, weights) + bias
        
        if self.n_classes_ == 2:
            probs = self._sigmoid(logits)
            loss = -np.mean(y_encoded * np.log(probs + 1e-15) + 
                           (1 - y_encoded) * np.log(1 - probs + 1e-15))
        else:
            probs = self._softmax(logits)
            loss = -np.mean(np.sum(y_encoded * np.log(probs + 1e-15), axis=1))
        
        # Add L2 regularization (complexity penalty)
        l2_penalty = self.complexity_penalty * np.sum(weights ** 2)
        return loss + l2_penalty
    
    def _compute_gradients(self, X, y_encoded):
        """Compute gradients for weights and bias."""
        n_samples = X.shape[0]
        logits = np.dot(X, self.weights_) + self.bias_
        
        if self.n_classes_ == 2:
            probs = self._sigmoid(logits)
        else:
            probs = self._softmax(logits)
        
        error = probs - y_encoded
        
        grad_weights = np.dot(X.T, error) / n_samples
        grad_weights += 2 * self.complexity_penalty * self.weights_
        grad_bias = np.mean(error, axis=0, keepdims=True)
        
        return grad_weights, grad_bias
    
    def _update_gradient_history(self, grad_weights, iteration):
        """Track gradient magnitudes and identify pruning candidates."""
        # Compute L2 norm of gradients for each feature
        feature_grad_norms = np.linalg.norm(grad_weights, axis=1)
        
        # Update low gradient counter
        low_grad_mask = feature_grad_norms < self.gradient_threshold
        self.low_gradient_counter_[low_grad_mask] += 1
        self.low_gradient_counter_[~low_grad_mask] = 0
        
        # Prune features with consistently low gradients
        if iteration % self.prune_check_interval == 0:
            prune_mask = (self.low_gradient_counter_ >= self.prune_patience) & self.active_features_
            if np.any(prune_mask):
                self.active_features_[prune_mask] = False
                self.weights_[prune_mask, :] = 0
                self.low_gradient_counter_[prune_mask] = 0
                return np.sum(prune_mask)
        
        return 0
    
    def fit(self, X, y):
        """
        Fit the classifier with adaptive pruning.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        n_samples, n_features = X_scaled.shape
        
        # Encode labels
        if self.n_classes_ == 2:
            y_encoded = (y == self.classes_[1]).astype(float).reshape(-1, 1)
        else:
            y_encoded = np.zeros((n_samples, self.n_classes_))
            for idx, cls in enumerate(self.classes_):
                y_encoded[y == cls, idx] = 1
        
        # Initialize weights
        self._initialize_weights(n_features, self.n_classes_ if self.n_classes_ > 2 else 1)
        
        # Training loop
        prev_loss = float('inf')
        self.loss_history_ = []
        self.pruned_features_history_ = []
        
        for iteration in range(self.max_iter):
            # Only use active features
            X_active = X_scaled[:, self.active_features_]
            weights_active = self.weights_[self.active_features_, :]
            
            # Compute loss
            loss = self._compute_loss(X_active, y_encoded, weights_active, self.bias_)
            self.loss_history_.append(loss)
            
            # Compute gradients on full feature space
            grad_weights, grad_bias = self._compute_gradients(X_scaled, y_encoded)
            
            # Update gradient history and prune
            n_pruned = self._update_gradient_history(grad_weights, iteration)
            if n_pruned > 0:
                self.pruned_features_history_.append((iteration, n_pruned))
            
            # Update weights and bias
            self.weights_ -= self.learning_rate * grad_weights
            self.bias_ -= self.learning_rate * grad_bias
            
            # Zero out pruned weights
            self.weights_[~self.active_features_, :] = 0
            
            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                break
            
            prev_loss = loss
        
        self.n_iter_ = iteration + 1
        self.n_active_features_ = np.sum(self.active_features_)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ['weights_', 'bias_', 'scaler_'])
        X = check_array(X)
        X_scaled = self.scaler_.transform(X)
        
        logits = np.dot(X_scaled, self.weights_) + self.bias_
        
        if self.n_classes_ == 2:
            probs = self._sigmoid(logits)
            return np.hstack([1 - probs, probs])
        else:
            return self._softmax(logits)
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def get_active_features(self):
        """
        Get indices of active (non-pruned) features.
        
        Returns
        -------
        active_indices : array
            Indices of active features.
        """
        check_is_fitted(self, ['active_features_'])
        return np.where(self.active_features_)[0]