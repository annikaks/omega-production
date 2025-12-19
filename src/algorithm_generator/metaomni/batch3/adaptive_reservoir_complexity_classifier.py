import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import warnings


class AdaptiveReservoirComplexityClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses dynamic reservoir sparsity adapted based on input complexity
    measured by local Kolmogorov estimates.
    
    Parameters
    ----------
    n_reservoir : int, default=500
        Number of reservoir nodes
    spectral_radius : float, default=0.9
        Spectral radius of the reservoir weight matrix
    input_scaling : float, default=1.0
        Scaling factor for input weights
    leak_rate : float, default=0.3
        Leaking rate for reservoir states
    min_sparsity : float, default=0.1
        Minimum sparsity level (fraction of connections to keep)
    max_sparsity : float, default=0.9
        Maximum sparsity level (fraction of connections to keep)
    k_neighbors : int, default=5
        Number of neighbors for local Kolmogorov complexity estimation
    ridge_param : float, default=1e-6
        Ridge regression regularization parameter
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_reservoir=500, spectral_radius=0.9, input_scaling=1.0,
                 leak_rate=0.3, min_sparsity=0.1, max_sparsity=0.9,
                 k_neighbors=5, ridge_param=1e-6, random_state=None):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self.k_neighbors = k_neighbors
        self.ridge_param = ridge_param
        self.random_state = random_state
    
    def _estimate_local_kolmogorov_complexity(self, X, reference_X=None):
        """
        Estimate local Kolmogorov complexity using k-nearest neighbor distances.
        Higher values indicate higher complexity.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to estimate complexity for
        reference_X : array-like of shape (n_reference, n_features), optional
            Reference samples to compute distances against. If None, uses X itself.
        
        Returns
        -------
        complexities : array of shape (n_samples,)
            Normalized complexity estimates in [0, 1]
        """
        if reference_X is None:
            reference_X = X
        
        n_samples = X.shape[0]
        complexities = np.zeros(n_samples)
        
        # Compute pairwise distances
        distances = cdist(X, reference_X, metric='euclidean')
        
        for i in range(n_samples):
            # Get k nearest neighbors (excluding self if X == reference_X)
            dists = distances[i]
            if reference_X is X:
                # Exclude self (distance = 0)
                sorted_indices = np.argsort(dists)[1:self.k_neighbors + 1]
            else:
                sorted_indices = np.argsort(dists)[:self.k_neighbors]
            
            neighbor_dists = dists[sorted_indices]
            
            # Estimate complexity as the average distance to k-nearest neighbors
            # weighted by the coefficient of variation
            mean_dist = np.mean(neighbor_dists)
            std_dist = np.std(neighbor_dists)
            
            # Higher variance relative to mean indicates higher local complexity
            # Add small epsilon to avoid division by zero
            complexities[i] = mean_dist * (1 + std_dist / (mean_dist + 1e-10))
        
        # Normalize to [0, 1]
        min_c = np.min(complexities)
        max_c = np.max(complexities)
        
        if max_c > min_c:
            complexities = (complexities - min_c) / (max_c - min_c)
        else:
            complexities = np.ones_like(complexities) * 0.5
        
        return complexities
    
    def _compute_adaptive_sparsity(self, complexity):
        """
        Compute sparsity level based on input complexity.
        Higher complexity -> lower sparsity (more connections).
        
        Parameters
        ----------
        complexity : float or array-like
            Complexity measure(s) in [0, 1]
        
        Returns
        -------
        sparsity : float or array-like
            Sparsity level(s) in [min_sparsity, max_sparsity]
        """
        # Invert complexity: high complexity needs more connections (lower sparsity)
        sparsity = self.max_sparsity - complexity * (self.max_sparsity - self.min_sparsity)
        return sparsity
    
    def _initialize_reservoir(self, n_features):
        """Initialize reservoir weights."""
        rng = np.random.RandomState(self.random_state)
        
        # Input weights
        self.W_in_ = rng.uniform(-self.input_scaling, self.input_scaling,
                                  (self.n_reservoir, n_features))
        
        # Initial reservoir weights (dense, will be sparsified adaptively)
        W_reservoir = rng.randn(self.n_reservoir, self.n_reservoir)
        
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W_reservoir)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            W_reservoir = W_reservoir * (self.spectral_radius / current_radius)
        
        self.W_reservoir_base_ = W_reservoir
        
    def _apply_dynamic_sparsity(self, W_reservoir, sparsity_level):
        """
        Apply sparsity to reservoir matrix based on complexity.
        
        Parameters
        ----------
        W_reservoir : array-like of shape (n_reservoir, n_reservoir)
            Base reservoir weight matrix
        sparsity_level : float
            Fraction of connections to remove (0 = keep all, 1 = remove all)
        
        Returns
        -------
        W_sparse : array of shape (n_reservoir, n_reservoir)
            Sparsified reservoir matrix
        """
        # Keep only top (1 - sparsity_level) fraction of weights by magnitude
        n_keep = int((1 - sparsity_level) * W_reservoir.size)
        n_keep = max(1, n_keep)  # Keep at least one connection
        
        flat_weights = np.abs(W_reservoir.flatten())
        
        if n_keep >= len(flat_weights):
            return W_reservoir.copy()
        
        # Find threshold value
        threshold = np.partition(flat_weights, -n_keep)[-n_keep]
        
        # Create mask and apply
        mask = np.abs(W_reservoir) >= threshold
        W_sparse = W_reservoir * mask
        
        return W_sparse
    
    def _compute_reservoir_states(self, X, sparsity_levels):
        """
        Compute reservoir states with adaptive sparsity.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data
        sparsity_levels : array-like of shape (n_samples,)
            Sparsity level for each sample
        
        Returns
        -------
        states : array of shape (n_samples, n_reservoir)
            Reservoir states
        """
        n_samples = X.shape[0]
        states = np.zeros((n_samples, self.n_reservoir))
        
        state = np.zeros(self.n_reservoir)
        
        for i in range(n_samples):
            # Apply adaptive sparsity for this sample
            W_adapted = self._apply_dynamic_sparsity(self.W_reservoir_base_, sparsity_levels[i])
            
            # Update reservoir state
            input_activation = self.W_in_ @ X[i]
            reservoir_activation = W_adapted @ state
            
            # Ensure reservoir_activation is a 1D array
            if hasattr(reservoir_activation, 'toarray'):
                reservoir_activation = reservoir_activation.toarray().flatten()
            elif reservoir_activation.ndim > 1:
                reservoir_activation = reservoir_activation.flatten()
            
            state = (1 - self.leak_rate) * state + \
                    self.leak_rate * np.tanh(input_activation + reservoir_activation)
            
            states[i] = state
        
        return states
    
    def fit(self, X, y):
        """
        Fit the adaptive reservoir classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns
        -------
        self : object
            Returns self
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Store training data for complexity estimation during prediction
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.X_train_ = X_scaled
        
        # Estimate local Kolmogorov complexity
        self.train_complexities_ = self._estimate_local_kolmogorov_complexity(X_scaled)
        
        # Compute adaptive sparsity levels
        self.train_sparsity_levels_ = self._compute_adaptive_sparsity(self.train_complexities_)
        
        # Initialize reservoir
        self._initialize_reservoir(self.n_features_in_)
        
        # Compute reservoir states with adaptive sparsity
        reservoir_states = self._compute_reservoir_states(X_scaled, self.train_sparsity_levels_)
        
        # Train output weights using ridge regression
        # For classification, we use one-hot encoding
        n_classes = len(self.classes_)
        Y_encoded = np.zeros((len(y), n_classes))
        for i, label in enumerate(y):
            Y_encoded[i, np.where(self.classes_ == label)[0][0]] = 1
        
        # Ridge regression
        self.W_out_ = np.linalg.solve(
            reservoir_states.T @ reservoir_states + self.ridge_param * np.eye(self.n_reservoir),
            reservoir_states.T @ Y_encoded
        )
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        
        # Standardize input
        X_scaled = self.scaler_.transform(X)
        
        # Estimate complexity for test samples relative to training data
        test_complexities = self._estimate_local_kolmogorov_complexity(X_scaled, self.X_train_)
        
        # Compute adaptive sparsity levels
        test_sparsity_levels = self._compute_adaptive_sparsity(test_complexities)
        
        # Compute reservoir states
        reservoir_states = self._compute_reservoir_states(X_scaled, test_sparsity_levels)
        
        # Predict
        y_pred_scores = reservoir_states @ self.W_out_
        
        # Convert to class labels (argmax)
        y_pred_indices = np.argmax(y_pred_scores, axis=1)
        y_pred = self.classes_[y_pred_indices]
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
        
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        
        # Standardize input
        X_scaled = self.scaler_.transform(X)
        
        # Estimate complexity for test samples
        test_complexities = self._estimate_local_kolmogorov_complexity(X_scaled, self.X_train_)
        
        # Compute adaptive sparsity levels
        test_sparsity_levels = self._compute_adaptive_sparsity(test_complexities)
        
        # Compute reservoir states
        reservoir_states = self._compute_reservoir_states(X_scaled, test_sparsity_levels)
        
        # Predict scores
        y_pred_scores = reservoir_states @ self.W_out_
        
        # Convert to probabilities using softmax
        exp_scores = np.exp(y_pred_scores - np.max(y_pred_scores, axis=1, keepdims=True))
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return proba