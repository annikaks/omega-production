import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import rbf_kernel

class DimAwareConnector(BaseEstimator, ClassifierMixin):
    def __init__(self, n_connections=10, gamma='scale', random_state=None):
        self.n_connections = n_connections
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_in_ = X.shape[1]
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Generate random connections
        self.connections_ = rng.randn(self.n_connections, self.n_features_in_)
        
        # Compute feature interactions
        X_interactions = self._compute_interactions(X)
        
        # Compute RBF kernel
        if self.gamma == 'scale':
            gamma = 1 / (self.n_features_in_ * X.var())
        elif self.gamma == 'auto':
            gamma = 1 / self.n_features_in_
        else:
            gamma = self.gamma
        
        self.kernel_matrix_ = rbf_kernel(X_interactions, self.connections_, gamma=gamma)
        
        # Compute weights using pseudo-inverse
        self.weights_ = np.linalg.pinv(self.kernel_matrix_) @ y
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Compute feature interactions
        X_interactions = self._compute_interactions(X)
        
        # Compute RBF kernel
        if self.gamma == 'scale':
            gamma = 1 / (self.n_features_in_ * X.var())
        elif self.gamma == 'auto':
            gamma = 1 / self.n_features_in_
        else:
            gamma = self.gamma
        
        kernel_matrix = rbf_kernel(X_interactions, self.connections_, gamma=gamma)
        
        # Make predictions
        y_pred = kernel_matrix @ self.weights_
        
        return np.round(y_pred).astype(int)

    def _compute_interactions(self, X):
        # Compute pairwise feature interactions
        n_samples, n_features = X.shape
        interactions = np.zeros((n_samples, n_features * (n_features + 1) // 2))
        
        idx = 0
        for i in range(n_features):
            for j in range(i, n_features):
                interactions[:, idx] = X[:, i] * X[:, j]
                idx += 1
        
        return np.hstack((X, interactions))