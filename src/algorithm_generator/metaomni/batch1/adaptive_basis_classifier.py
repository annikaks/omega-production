import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.special import expit, softmax


class AdaptiveBasisClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that replaces sigmoid with a learnable mixture of basis functions
    that adapts complexity to different data regions.
    
    The activation is modeled as: f(x) = sum_i w_i * phi_i(x)
    where phi_i are basis functions (e.g., Gaussians, polynomials) and w_i are learned weights.
    
    Supports both binary and multi-class classification.
    
    Parameters
    ----------
    n_basis : int, default=5
        Number of basis functions in the mixture
    basis_type : str, default='gaussian'
        Type of basis functions ('gaussian', 'polynomial', 'sigmoid_mixture')
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    max_iter : int, default=1000
        Maximum number of iterations for optimization
    reg_lambda : float, default=0.01
        L2 regularization parameter
    tol : float, default=1e-4
        Tolerance for optimization convergence
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_basis=5, basis_type='gaussian', learning_rate=0.01,
                 max_iter=1000, reg_lambda=0.01, tol=1e-4, random_state=None):
        self.n_basis = n_basis
        self.basis_type = basis_type
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.reg_lambda = reg_lambda
        self.tol = tol
        self.random_state = random_state
    
    def _adaptive_activation(self, z, class_idx=0):
        """Apply learnable mixture of basis functions."""
        # z is the linear combination: X @ weights
        activations = np.zeros_like(z)
        
        if self.basis_type == 'gaussian':
            for i in range(self.n_basis):
                center = self.activation_centers_[class_idx, i]
                width = self.activation_widths_[class_idx, i]
                activations += self.basis_weights_[class_idx, i] * np.exp(-0.5 * ((z - center) / (width + 1e-8))**2)
        
        elif self.basis_type == 'sigmoid_mixture':
            for i in range(self.n_basis):
                slope = self.activation_slopes_[class_idx, i]
                shift = self.activation_shifts_[class_idx, i]
                activations += self.basis_weights_[class_idx, i] * expit(slope * (z - shift))
        
        elif self.basis_type == 'polynomial':
            for i in range(self.n_basis):
                degree = i + 1
                activations += self.basis_weights_[class_idx, i] * (z ** degree)
            activations = expit(activations)  # Final sigmoid for probability
        
        return np.clip(activations, 1e-10, 1 - 1e-10)
    
    def _initialize_activation_params(self, X, y, class_idx):
        """Initialize parameters for the adaptive activation function."""
        rng = np.random.RandomState(self.random_state)
        
        # Compute initial linear scores
        z = X @ self.coef_[class_idx] + self.intercept_[class_idx]
        
        if self.basis_type == 'gaussian':
            # Use quantiles of z for centers
            try:
                centers = np.percentile(z, np.linspace(10, 90, self.n_basis))
            except:
                centers = np.linspace(z.min(), z.max(), self.n_basis)
            self.activation_centers_[class_idx] = centers
            self.activation_widths_[class_idx] = np.ones(self.n_basis) * (np.std(z) + 1e-8)
            self.basis_weights_[class_idx] = np.ones(self.n_basis) / self.n_basis
        
        elif self.basis_type == 'sigmoid_mixture':
            self.activation_slopes_[class_idx] = rng.randn(self.n_basis) * 0.5 + 1.0
            try:
                shifts = np.percentile(z, np.linspace(10, 90, self.n_basis))
            except:
                shifts = np.linspace(z.min(), z.max(), self.n_basis)
            self.activation_shifts_[class_idx] = shifts
            self.basis_weights_[class_idx] = np.ones(self.n_basis) / self.n_basis
        
        elif self.basis_type == 'polynomial':
            self.basis_weights_[class_idx] = np.ones(self.n_basis) / self.n_basis
    
    def fit(self, X, y):
        """
        Fit the adaptive basis classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        n_samples, n_features = X.shape
        
        # Normalize features
        self.X_mean_ = np.mean(X, axis=0)
        self.X_std_ = np.std(X, axis=0) + 1e-8
        X_normalized = (X - self.X_mean_) / self.X_std_
        
        # Initialize parameters for each class (one-vs-rest)
        rng = np.random.RandomState(self.random_state)
        self.coef_ = rng.randn(self.n_classes_, n_features) * 0.01
        self.intercept_ = np.zeros(self.n_classes_)
        
        # Initialize activation parameters
        if self.basis_type == 'gaussian':
            self.activation_centers_ = np.zeros((self.n_classes_, self.n_basis))
            self.activation_widths_ = np.ones((self.n_classes_, self.n_basis))
        elif self.basis_type == 'sigmoid_mixture':
            self.activation_slopes_ = np.ones((self.n_classes_, self.n_basis))
            self.activation_shifts_ = np.zeros((self.n_classes_, self.n_basis))
        
        self.basis_weights_ = np.ones((self.n_classes_, self.n_basis)) / self.n_basis
        
        # Train one classifier per class (one-vs-rest)
        for class_idx, class_label in enumerate(self.classes_):
            y_binary = (y == class_label).astype(float)
            
            # Initialize activation parameters for this class
            self._initialize_activation_params(X_normalized, y_binary, class_idx)
            
            # Optimization using gradient descent
            prev_loss = float('inf')
            
            for iteration in range(self.max_iter):
                # Forward pass
                z = X_normalized @ self.coef_[class_idx] + self.intercept_[class_idx]
                y_pred = self._adaptive_activation(z, class_idx)
                
                # Compute loss (binary cross-entropy + L2 regularization)
                loss = -np.mean(y_binary * np.log(y_pred) + (1 - y_binary) * np.log(1 - y_pred))
                loss += 0.5 * self.reg_lambda * np.sum(self.coef_[class_idx]**2)
                
                # Check convergence
                if abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss
                
                # Backward pass (gradient computation)
                error = y_pred - y_binary
                
                # Update linear coefficients
                grad_coef = X_normalized.T @ error / n_samples + self.reg_lambda * self.coef_[class_idx]
                grad_intercept = np.mean(error)
                
                self.coef_[class_idx] -= self.learning_rate * grad_coef
                self.intercept_[class_idx] -= self.learning_rate * grad_intercept
                
                # Update basis weights (simplified gradient)
                if self.basis_type in ['gaussian', 'sigmoid_mixture']:
                    basis_grad = np.zeros(self.n_basis)
                    for i in range(self.n_basis):
                        if self.basis_type == 'gaussian':
                            center = self.activation_centers_[class_idx, i]
                            width = self.activation_widths_[class_idx, i]
                            basis_contrib = np.exp(-0.5 * ((z - center) / (width + 1e-8))**2)
                        else:  # sigmoid_mixture
                            slope = self.activation_slopes_[class_idx, i]
                            shift = self.activation_shifts_[class_idx, i]
                            basis_contrib = expit(slope * (z - shift))
                        
                        basis_grad[i] = np.mean(error * basis_contrib)
                    
                    self.basis_weights_[class_idx] -= self.learning_rate * basis_grad
                    self.basis_weights_[class_idx] = np.maximum(self.basis_weights_[class_idx], 0)
                    weight_sum = np.sum(self.basis_weights_[class_idx])
                    if weight_sum > 1e-8:
                        self.basis_weights_[class_idx] /= weight_sum
                    else:
                        self.basis_weights_[class_idx] = np.ones(self.n_basis) / self.n_basis
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self)
        X = check_array(X)
        
        X_normalized = (X - self.X_mean_) / self.X_std_
        n_samples = X.shape[0]
        
        # Compute scores for each class
        scores = np.zeros((n_samples, self.n_classes_))
        for class_idx in range(self.n_classes_):
            z = X_normalized @ self.coef_[class_idx] + self.intercept_[class_idx]
            scores[:, class_idx] = self._adaptive_activation(z, class_idx)
        
        # Normalize to probabilities using softmax
        if self.n_classes_ == 2:
            # For binary classification, use the standard approach
            proba = np.zeros((n_samples, 2))
            proba[:, 1] = scores[:, 1]
            proba[:, 0] = 1 - proba[:, 1]
        else:
            # For multi-class, use softmax on the scores
            proba = softmax(scores, axis=1)
        
        return proba
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]