import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.optimize import minimize


class SparseDiscriminantClassifier(BaseEstimator, ClassifierMixin):
    """
    Sparse Linear Discriminant Analysis Classifier with L1 regularization.
    
    This classifier performs discriminant analysis while applying sparse regularization
    on the discriminant directions to automatically select relevant features while
    maintaining interpretability.
    
    Parameters
    ----------
    alpha : float, default=0.1
        Regularization strength for L1 penalty on discriminant directions.
        Larger values lead to more sparsity.
    
    n_components : int, default=None
        Number of discriminant components to use. If None, uses min(n_classes-1, n_features).
    
    tol : float, default=1e-4
        Tolerance for optimization convergence.
    
    max_iter : int, default=1000
        Maximum number of iterations for optimization.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    
    coef_ : ndarray of shape (n_components, n_features)
        Sparse discriminant directions (coefficients).
    
    intercept_ : ndarray of shape (n_classes,)
        Intercept terms for each class.
    
    feature_importances_ : ndarray of shape (n_features,)
        Feature importance scores based on coefficient magnitudes.
    """
    
    def __init__(self, alpha=0.1, n_components=None, tol=1e-4, max_iter=1000):
        self.alpha = alpha
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
    
    def _compute_class_statistics(self, X, y):
        """Compute class means and covariance matrices."""
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Compute class means
        class_means = np.zeros((n_classes, n_features))
        class_counts = np.zeros(n_classes)
        
        for idx, cls in enumerate(self.classes_):
            mask = (y == cls)
            class_means[idx] = X[mask].mean(axis=0)
            class_counts[idx] = mask.sum()
        
        # Overall mean
        overall_mean = X.mean(axis=0)
        
        # Within-class scatter matrix
        Sw = np.zeros((n_features, n_features))
        for idx, cls in enumerate(self.classes_):
            mask = (y == cls)
            X_centered = X[mask] - class_means[idx]
            Sw += X_centered.T @ X_centered
        
        # Between-class scatter matrix
        Sb = np.zeros((n_features, n_features))
        for idx, cls in enumerate(self.classes_):
            diff = (class_means[idx] - overall_mean).reshape(-1, 1)
            Sb += class_counts[idx] * (diff @ diff.T)
        
        return class_means, Sw, Sb, overall_mean
    
    def _optimize_sparse_direction(self, Sw, Sb, n_components):
        """Optimize sparse discriminant directions using L1 regularization."""
        n_features = Sw.shape[0]
        
        # Add regularization to Sw for numerical stability
        Sw_reg = Sw + 1e-6 * np.eye(n_features)
        
        # Initialize with standard LDA solution
        try:
            eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(Sw_reg) @ Sb)
            idx = np.argsort(eigvals)[::-1]
            W_init = eigvecs[:, idx[:n_components]]
        except:
            W_init = np.random.randn(n_features, n_components) * 0.01
        
        # Optimize each component with L1 regularization
        W_sparse = np.zeros((n_components, n_features))
        
        for comp in range(n_components):
            w_init = W_init[:, comp]
            
            def objective(w):
                w = w.reshape(-1, 1)
                # Maximize between-class variance, minimize within-class variance
                between = w.T @ Sb @ w
                within = w.T @ Sw_reg @ w
                ratio = -between / (within + 1e-8)
                # Add L1 penalty
                l1_penalty = self.alpha * np.sum(np.abs(w))
                return ratio.item() + l1_penalty
            
            def gradient(w):
                w = w.reshape(-1, 1)
                between = w.T @ Sb @ w
                within = w.T @ Sw_reg @ w
                
                grad_between = 2 * Sb @ w
                grad_within = 2 * Sw_reg @ w
                
                grad_ratio = -(grad_between * within - between * grad_within) / (within**2 + 1e-8)
                grad_l1 = self.alpha * np.sign(w)
                
                return (grad_ratio + grad_l1).flatten()
            
            # Optimize
            result = minimize(
                objective,
                w_init,
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': self.max_iter, 'ftol': self.tol}
            )
            
            W_sparse[comp] = result.x
            
            # Apply soft thresholding for sparsity
            threshold = self.alpha * 0.1
            W_sparse[comp][np.abs(W_sparse[comp]) < threshold] = 0
        
        return W_sparse
    
    def fit(self, X, y):
        """
        Fit the Sparse Discriminant Classifier.
        
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
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        
        # Store classes
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Determine number of components
        if self.n_components is None:
            self.n_components_ = min(n_classes - 1, n_features)
        else:
            self.n_components_ = min(self.n_components, n_classes - 1, n_features)
        
        # Compute class statistics
        self.class_means_, Sw, Sb, self.overall_mean_ = self._compute_class_statistics(X, y_encoded)
        
        # Optimize sparse discriminant directions
        self.coef_ = self._optimize_sparse_direction(Sw, Sb, self.n_components_)
        
        # Compute feature importances
        self.feature_importances_ = np.sum(np.abs(self.coef_), axis=0)
        self.feature_importances_ /= (self.feature_importances_.sum() + 1e-10)
        
        # Compute intercepts for classification
        self.intercept_ = np.zeros(n_classes)
        
        # Project class means to discriminant space
        class_projections = self.class_means_ @ self.coef_.T
        
        # Store for prediction
        self.class_projections_ = class_projections
        
        return self
    
    def _project(self, X):
        """Project data onto discriminant directions."""
        return X @ self.coef_.T
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ['coef_', 'class_projections_'])
        X = check_array(X)
        
        # Project test data
        X_proj = self._project(X)
        
        # Compute distances to class centroids in projected space
        distances = np.zeros((X.shape[0], len(self.classes_)))
        
        for idx in range(len(self.classes_)):
            diff = X_proj - self.class_projections_[idx]
            distances[:, idx] = np.sum(diff**2, axis=1)
        
        # Predict class with minimum distance
        y_pred_encoded = np.argmin(distances, axis=1)
        
        # Decode labels
        y_pred = self.label_encoder_.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ['coef_', 'class_projections_'])
        X = check_array(X)
        
        # Project test data
        X_proj = self._project(X)
        
        # Compute distances to class centroids
        distances = np.zeros((X.shape[0], len(self.classes_)))
        
        for idx in range(len(self.classes_)):
            diff = X_proj - self.class_projections_[idx]
            distances[:, idx] = np.sum(diff**2, axis=1)
        
        # Convert distances to probabilities using softmax
        neg_distances = -distances
        exp_distances = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
        proba = exp_distances / np.sum(exp_distances, axis=1, keepdims=True)
        
        return proba
    
    def get_support(self, threshold=1e-6):
        """
        Get mask of selected features.
        
        Parameters
        ----------
        threshold : float, default=1e-6
            Threshold for considering a feature as selected.
        
        Returns
        -------
        support : ndarray of shape (n_features,)
            Boolean mask of selected features.
        """
        check_is_fitted(self, ['feature_importances_'])
        return self.feature_importances_ > threshold