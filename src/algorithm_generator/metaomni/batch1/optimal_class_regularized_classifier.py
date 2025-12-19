import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.linalg import pinv


class OptimalClassRegularizedClassifier(BaseEstimator, ClassifierMixin):
    """
    Optimal Class-Specific Regularized Classifier with closed-form solution.
    
    This classifier solves a multi-objective optimization problem that minimizes
    within-class variance while maximizing between-class separation. It derives
    class-specific regularization parameters through eigenvalue decomposition
    of within-class and between-class scatter matrices.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Global regularization strength parameter.
    
    beta : float, default=0.1
        Class-specific regularization adaptation rate.
    
    shrinkage : float, default=1e-6
        Shrinkage parameter for numerical stability.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    
    class_means_ : ndarray of shape (n_classes, n_features)
        Mean vector for each class.
    
    class_regularizers_ : ndarray of shape (n_classes,)
        Optimal regularization parameter for each class.
    
    projection_matrix_ : ndarray of shape (n_features, n_features)
        Optimal projection matrix derived from the closed-form solution.
    
    global_mean_ : ndarray of shape (n_features,)
        Global mean of all training samples.
    """
    
    def __init__(self, alpha=1.0, beta=0.1, shrinkage=1e-6):
        self.alpha = alpha
        self.beta = beta
        self.shrinkage = shrinkage
    
    def fit(self, X, y):
        """
        Fit the classifier by deriving optimal class-specific regularization.
        
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
        self.n_features_in_ = X.shape[1]
        
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # Compute global mean
        self.global_mean_ = np.mean(X, axis=0)
        
        # Initialize class-specific statistics
        self.class_means_ = np.zeros((n_classes, n_features))
        class_counts = np.zeros(n_classes)
        
        # Compute within-class scatter matrix (S_W) and between-class scatter (S_B)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        
        class_variances = np.zeros(n_classes)
        
        for idx, cls in enumerate(self.classes_):
            # Get samples for this class
            X_cls = X[y == cls]
            class_counts[idx] = X_cls.shape[0]
            
            # Compute class mean
            self.class_means_[idx] = np.mean(X_cls, axis=0)
            
            # Within-class scatter for this class
            X_centered = X_cls - self.class_means_[idx]
            S_W += X_centered.T @ X_centered
            
            # Between-class scatter contribution
            mean_diff = (self.class_means_[idx] - self.global_mean_).reshape(-1, 1)
            S_B += class_counts[idx] * (mean_diff @ mean_diff.T)
            
            # Compute class variance (trace of covariance)
            class_variances[idx] = np.trace(X_centered.T @ X_centered) / class_counts[idx]
        
        # Normalize scatter matrices
        S_W /= n_samples
        S_B /= n_samples
        
        # Add shrinkage for numerical stability
        S_W += self.shrinkage * np.eye(n_features)
        
        # Derive optimal class-specific regularization parameters
        # Based on the ratio of within-class variance to total variance
        total_variance = np.sum(class_variances)
        if total_variance > 0:
            self.class_regularizers_ = self.alpha * (1.0 + self.beta * class_variances / total_variance)
        else:
            self.class_regularizers_ = np.full(n_classes, self.alpha)
        
        # Solve generalized eigenvalue problem: S_B * v = lambda * S_W * v
        # This gives us the optimal projection directions
        try:
            S_W_inv = pinv(S_W)
            M = S_W_inv @ S_B
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(M)
            
            # Sort by eigenvalues (descending)
            idx_sorted = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx_sorted]
            eigenvectors = eigenvectors[:, idx_sorted]
            
            # Keep top components (up to n_classes - 1)
            n_components = min(n_classes - 1, n_features)
            self.projection_matrix_ = eigenvectors[:, :n_components].real
            
            # Compute class-specific projection weights
            self.class_weights_ = np.zeros((n_classes, n_components))
            for idx, cls in enumerate(self.classes_):
                # Weight by inverse of regularization (less regularization = more weight)
                weight = 1.0 / self.class_regularizers_[idx]
                self.class_weights_[idx] = weight * np.ones(n_components)
            
            # Normalize weights
            self.class_weights_ /= np.sum(self.class_weights_, axis=1, keepdims=True)
            
        except np.linalg.LinAlgError:
            # Fallback: use identity projection
            self.projection_matrix_ = np.eye(n_features)
            self.class_weights_ = np.ones((n_classes, n_features)) / n_features
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # Project data
        X_projected = X @ self.projection_matrix_
        
        # Compute regularized distances to each class mean
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        distances = np.zeros((n_samples, n_classes))
        
        for idx, cls in enumerate(self.classes_):
            # Project class mean
            class_mean_projected = self.class_means_[idx] @ self.projection_matrix_
            
            # Compute weighted Euclidean distance with class-specific regularization
            diff = X_projected - class_mean_projected
            
            # Apply class-specific weights
            weighted_diff = diff * self.class_weights_[idx]
            
            # Regularized distance
            distances[:, idx] = np.sum(weighted_diff ** 2, axis=1) * self.class_regularizers_[idx]
        
        # Predict class with minimum regularized distance
        y_pred = self.classes_[np.argmin(distances, axis=1)]
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # Project data
        X_projected = X @ self.projection_matrix_
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        distances = np.zeros((n_samples, n_classes))
        
        for idx, cls in enumerate(self.classes_):
            class_mean_projected = self.class_means_[idx] @ self.projection_matrix_
            diff = X_projected - class_mean_projected
            weighted_diff = diff * self.class_weights_[idx]
            distances[:, idx] = np.sum(weighted_diff ** 2, axis=1) * self.class_regularizers_[idx]
        
        # Convert distances to probabilities using softmax
        # Negative distances because smaller distance = higher probability
        exp_neg_dist = np.exp(-distances)
        proba = exp_neg_dist / np.sum(exp_neg_dist, axis=1, keepdims=True)
        
        return proba