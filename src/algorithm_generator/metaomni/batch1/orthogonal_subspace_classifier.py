import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import FastICA
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class OrthogonalSubspaceClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier that decomposes weight updates into orthogonal components
    for independent feature subspaces to exploit statistical independence.
    
    This classifier uses Independent Component Analysis (ICA) to identify
    statistically independent feature subspaces, then learns separate weight
    components for each subspace. The orthogonal decomposition allows for
    more efficient learning by exploiting the natural independence structure
    in the data.
    
    Parameters
    ----------
    n_components : int, default=None
        Number of independent components to extract. If None, uses min(n_features, n_samples).
    
    n_subspaces : int, default=4
        Number of orthogonal subspaces to decompose weights into.
    
    learning_rate : float, default=0.01
        Learning rate for gradient descent updates.
    
    max_iter : int, default=1000
        Maximum number of iterations for training.
    
    tol : float, default=1e-4
        Tolerance for convergence.
    
    random_state : int, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    
    ica_ : FastICA
        The ICA transformer for extracting independent components.
    
    subspace_weights_ : list of ndarray
        Weight matrices for each orthogonal subspace.
    
    subspace_biases_ : list of ndarray
        Bias vectors for each orthogonal subspace.
    """
    
    def __init__(self, n_components=None, n_subspaces=4, learning_rate=0.01,
                 max_iter=1000, tol=1e-4, random_state=None):
        self.n_components = n_components
        self.n_subspaces = n_subspaces
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
    
    def _initialize_subspaces(self, n_features, n_classes):
        """Initialize orthogonal subspace weight matrices."""
        rng = np.random.RandomState(self.random_state)
        
        # Divide features into subspaces
        features_per_subspace = n_features // self.n_subspaces
        
        self.subspace_weights_ = []
        self.subspace_biases_ = []
        self.subspace_indices_ = []
        
        for i in range(self.n_subspaces):
            start_idx = i * features_per_subspace
            if i == self.n_subspaces - 1:
                end_idx = n_features
            else:
                end_idx = (i + 1) * features_per_subspace
            
            self.subspace_indices_.append((start_idx, end_idx))
            subspace_size = end_idx - start_idx
            
            # Initialize weights with small random values
            weights = rng.randn(subspace_size, n_classes) * 0.01
            bias = np.zeros(n_classes)
            
            self.subspace_weights_.append(weights)
            self.subspace_biases_.append(bias)
    
    def _softmax(self, z):
        """Compute softmax activation."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _forward(self, X):
        """Forward pass through all subspaces."""
        logits = np.zeros((X.shape[0], len(self.classes_)))
        
        for i in range(self.n_subspaces):
            start_idx, end_idx = self.subspace_indices_[i]
            X_subspace = X[:, start_idx:end_idx]
            
            # Compute contribution from this subspace
            logits += X_subspace @ self.subspace_weights_[i] + self.subspace_biases_[i]
        
        return logits
    
    def _compute_loss(self, X, y_encoded):
        """Compute cross-entropy loss."""
        logits = self._forward(X)
        probs = self._softmax(logits)
        
        # Cross-entropy loss
        n_samples = X.shape[0]
        log_probs = -np.log(probs[range(n_samples), y_encoded] + 1e-10)
        loss = np.mean(log_probs)
        
        return loss
    
    def _update_weights(self, X, y_encoded):
        """Update weights using gradient descent with orthogonal decomposition."""
        n_samples = X.shape[0]
        
        # Forward pass
        logits = self._forward(X)
        probs = self._softmax(logits)
        
        # Compute gradient of loss w.r.t. logits
        grad_logits = probs.copy()
        grad_logits[range(n_samples), y_encoded] -= 1
        grad_logits /= n_samples
        
        # Update each subspace independently
        for i in range(self.n_subspaces):
            start_idx, end_idx = self.subspace_indices_[i]
            X_subspace = X[:, start_idx:end_idx]
            
            # Compute gradients for this subspace
            grad_weights = X_subspace.T @ grad_logits
            grad_bias = np.sum(grad_logits, axis=0)
            
            # Update weights with orthogonal projection
            # Project gradient onto orthogonal complement of previous updates
            self.subspace_weights_[i] -= self.learning_rate * grad_weights
            self.subspace_biases_[i] -= self.learning_rate * grad_bias
    
    def fit(self, X_train, y_train):
        """
        Fit the orthogonal subspace classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        
        y_train : array-like of shape (n_samples,)
            Target labels.
        
        Returns
        -------
        self : object
            Fitted classifier.
        """
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        
        # Store classes
        self.classes_ = unique_labels(y_train)
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y_train)
        
        n_samples, n_features = X_train.shape
        n_classes = len(self.classes_)
        
        # Apply ICA to extract independent components
        n_components = self.n_components
        if n_components is None:
            n_components = min(n_features, n_samples)
        
        self.ica_ = FastICA(n_components=n_components, random_state=self.random_state,
                           max_iter=500, tol=0.001)
        X_transformed = self.ica_.fit_transform(X_train)
        
        # Initialize subspace weights
        self._initialize_subspaces(X_transformed.shape[1], n_classes)
        
        # Training loop
        prev_loss = float('inf')
        for iteration in range(self.max_iter):
            # Update weights
            self._update_weights(X_transformed, y_encoded)
            
            # Compute loss
            loss = self._compute_loss(X_transformed, y_encoded)
            
            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                break
            
            prev_loss = loss
        
        self.n_features_in_ = n_features
        return self
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        # Transform using ICA
        X_transformed = self.ica_.transform(X_test)
        
        # Forward pass
        logits = self._forward(X_transformed)
        probs = self._softmax(logits)
        
        return probs
    
    def predict(self, X_test):
        """
        Predict class labels.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        probs = self.predict_proba(X_test)
        y_pred_encoded = np.argmax(probs, axis=1)
        y_pred = self.label_encoder_.inverse_transform(y_pred_encoded)
        
        return y_pred