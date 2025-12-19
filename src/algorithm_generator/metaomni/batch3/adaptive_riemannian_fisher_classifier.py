import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class AdaptiveRiemannianFisherClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Riemannian Fisher Classifier with manifold-aware metric learning.
    
    This classifier learns an adaptive Riemannian metric that compresses 
    high-curvature regions (smooth class boundaries) and expands low-curvature 
    regions (critical discrimination areas) by replacing the Euclidean Fisher 
    metric with a curvature-adaptive one.
    
    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep. If None, keeps all components.
    curvature_scale : float, default=1.0
        Scale parameter for curvature sensitivity. Higher values increase
        the effect of curvature on metric adaptation.
    n_neighbors : int, default=10
        Number of neighbors for local curvature estimation.
    regularization : float, default=1e-6
        Regularization parameter for numerical stability.
    metric_power : float, default=0.5
        Power to which the curvature-based metric is raised for adaptation.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    projection_ : ndarray of shape (n_features, n_components)
        The learned projection matrix in the Riemannian space.
    metric_tensor_ : ndarray of shape (n_features, n_features)
        The adaptive Riemannian metric tensor.
    class_means_ : ndarray of shape (n_classes, n_components)
        Class means in the projected space.
    """
    
    def __init__(self, n_components=None, curvature_scale=1.0, n_neighbors=10,
                 regularization=1e-6, metric_power=0.5):
        self.n_components = n_components
        self.curvature_scale = curvature_scale
        self.n_neighbors = n_neighbors
        self.regularization = regularization
        self.metric_power = metric_power
    
    def _estimate_local_curvature(self, X):
        """
        Estimate local curvature at each point using second-order differences.
        
        High curvature indicates smooth regions, low curvature indicates 
        critical discrimination regions.
        """
        n_samples, n_features = X.shape
        k = min(self.n_neighbors, n_samples - 1)
        
        # Compute pairwise distances
        distances = cdist(X, X, metric='euclidean')
        
        # Find k nearest neighbors for each point
        neighbor_indices = np.argsort(distances, axis=1)[:, 1:k+1]
        
        # Estimate curvature using local covariance structure
        curvatures = np.zeros(n_samples)
        
        for i in range(n_samples):
            neighbors = X[neighbor_indices[i]]
            centered = neighbors - X[i]
            
            # Compute local covariance
            cov = np.dot(centered.T, centered) / k
            
            # Curvature is related to the condition number of local covariance
            eigenvalues = np.linalg.eigvalsh(cov + self.regularization * np.eye(n_features))
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            
            # Use ratio of max to min eigenvalue as curvature measure
            curvatures[i] = np.max(eigenvalues) / (np.min(eigenvalues) + 1e-10)
        
        return curvatures
    
    def _compute_class_curvature_weights(self, X, y):
        """
        Compute per-class curvature weights for metric adaptation.
        """
        curvatures = self._estimate_local_curvature(X)
        
        # Normalize curvatures to [0, 1]
        curvatures = (curvatures - np.min(curvatures)) / (np.ptp(curvatures) + 1e-10)
        
        # Invert: high curvature -> compress (low weight), low curvature -> expand (high weight)
        weights = 1.0 / (1.0 + self.curvature_scale * curvatures)
        
        return weights
    
    def _compute_adaptive_fisher_metric(self, X, y, weights):
        """
        Compute the adaptive Fisher metric tensor with curvature-based weighting.
        """
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        
        # Compute global mean
        global_mean = np.mean(X, axis=0)
        
        # Compute weighted within-class scatter
        S_w = np.zeros((n_features, n_features))
        
        for c in classes:
            mask = (y == c)
            X_c = X[mask]
            weights_c = weights[mask]
            
            class_mean = np.mean(X_c, axis=0)
            centered = X_c - class_mean
            
            # Weight each sample by its curvature weight
            weighted_centered = centered * weights_c[:, np.newaxis]
            S_w += np.dot(weighted_centered.T, centered)
        
        S_w /= n_samples
        
        # Compute weighted between-class scatter
        S_b = np.zeros((n_features, n_features))
        
        for c in classes:
            mask = (y == c)
            n_c = np.sum(mask)
            class_mean = np.mean(X[mask], axis=0)
            
            # Average weight for this class
            avg_weight = np.mean(weights[mask])
            
            diff = (class_mean - global_mean).reshape(-1, 1)
            S_b += avg_weight * n_c * np.dot(diff, diff.T)
        
        S_b /= n_samples
        
        # Regularize
        S_w += self.regularization * np.eye(n_features)
        
        # The adaptive metric tensor emphasizes discriminative directions
        # Metric = S_w^(-1/2) * S_b * S_w^(-1/2)
        S_w_inv_sqrt = self._matrix_power(S_w, -0.5)
        metric_tensor = np.dot(np.dot(S_w_inv_sqrt, S_b), S_w_inv_sqrt)
        
        return metric_tensor, S_w, S_b
    
    def _matrix_power(self, M, power):
        """Compute matrix power using eigendecomposition."""
        eigenvalues, eigenvectors = eigh(M)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        return np.dot(eigenvectors, np.dot(np.diag(eigenvalues ** power), eigenvectors.T))
    
    def fit(self, X, y):
        """
        Fit the Adaptive Riemannian Fisher Classifier.
        
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
        
        # Store classes
        self.classes_ = unique_labels(y)
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # Set n_components
        if self.n_components is None:
            self.n_components_ = min(n_features, n_classes - 1)
        else:
            self.n_components_ = min(self.n_components, n_features, n_classes - 1)
        
        # Compute curvature-based weights
        weights = self._compute_class_curvature_weights(X, y_encoded)
        
        # Compute adaptive Fisher metric
        metric_tensor, S_w, S_b = self._compute_adaptive_fisher_metric(X, y_encoded, weights)
        
        # Store metric tensor
        self.metric_tensor_ = metric_tensor
        
        # Solve generalized eigenvalue problem: S_b * v = lambda * S_w * v
        eigenvalues, eigenvectors = eigh(S_b, S_w)
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep top n_components
        self.projection_ = eigenvectors[:, :self.n_components_]
        self.eigenvalues_ = eigenvalues[:self.n_components_]
        
        # Apply metric adaptation to projection
        metric_adapted = self._matrix_power(metric_tensor, self.metric_power)
        self.projection_ = np.dot(metric_adapted, self.projection_)
        
        # Normalize projection vectors
        for i in range(self.n_components_):
            norm = np.linalg.norm(self.projection_[:, i])
            if norm > 1e-10:
                self.projection_[:, i] /= norm
        
        # Project training data and compute class means
        X_projected = np.dot(X, self.projection_)
        
        self.class_means_ = np.zeros((n_classes, self.n_components_))
        for i, c in enumerate(self.classes_):
            mask = (y == c)
            self.class_means_[i] = np.mean(X_projected[mask], axis=0)
        
        return self
    
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
        # Check if fitted
        check_is_fitted(self, ['projection_', 'class_means_'])
        
        # Validate input
        X = check_array(X)
        
        # Project test data
        X_projected = np.dot(X, self.projection_)
        
        # Compute distances to class means in the Riemannian space
        distances = cdist(X_projected, self.class_means_, metric='euclidean')
        
        # Predict nearest class
        y_pred_encoded = np.argmin(distances, axis=1)
        
        # Decode labels
        y_pred = self.label_encoder_.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def transform(self, X):
        """
        Transform X to the learned Riemannian space.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self, ['projection_'])
        X = check_array(X)
        return np.dot(X, self.projection_)
    
    def decision_function(self, X):
        """
        Compute decision function (negative distances to class means).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        decision : ndarray of shape (n_samples, n_classes)
            Decision function values (negative distances).
        """
        check_is_fitted(self, ['projection_', 'class_means_'])
        X = check_array(X)
        
        X_projected = np.dot(X, self.projection_)
        distances = cdist(X_projected, self.class_means_, metric='euclidean')
        
        return -distances