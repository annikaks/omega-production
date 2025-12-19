import numpy as np
import zlib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict


class KolmogorovFractalClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses Kolmogorov complexity approximations via lossless 
    compression to weight fractal dimensions by algorithmic information content.
    
    This classifier combines:
    1. Kolmogorov complexity approximation using compression (zlib)
    2. Fractal dimension estimation (box-counting and correlation dimension)
    3. Algorithmic information content weighting
    
    Parameters
    ----------
    n_scales : int, default=10
        Number of scales for fractal dimension estimation
    compression_level : int, default=9
        Compression level for zlib (1-9)
    metric : str, default='weighted'
        Distance metric: 'weighted', 'kolmogorov', or 'fractal'
    alpha : float, default=0.5
        Weight balance between Kolmogorov and fractal features (0-1)
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes seen during fit
    X_train_ : ndarray of shape (n_samples, n_features)
        Training data
    y_train_ : ndarray of shape (n_samples,)
        Training labels
    kolmogorov_complexity_ : ndarray of shape (n_samples,)
        Kolmogorov complexity for each training sample
    fractal_dims_ : ndarray of shape (n_samples,)
        Fractal dimensions for each training sample
    """
    
    def __init__(self, n_scales=10, compression_level=9, metric='weighted', alpha=0.5):
        self.n_scales = n_scales
        self.compression_level = compression_level
        self.metric = metric
        self.alpha = alpha
    
    def _kolmogorov_complexity(self, x):
        """
        Approximate Kolmogorov complexity using lossless compression.
        
        Parameters
        ----------
        x : array-like
            Input data
            
        Returns
        -------
        float
            Normalized compression length as proxy for Kolmogorov complexity
        """
        # Convert to bytes
        x_bytes = x.tobytes()
        # Compress
        compressed = zlib.compress(x_bytes, level=self.compression_level)
        # Normalize by original length
        return len(compressed) / len(x_bytes)
    
    def _box_counting_dimension(self, x):
        """
        Estimate fractal dimension using box-counting method.
        
        Parameters
        ----------
        x : array-like of shape (n_features,)
            Input sample
            
        Returns
        -------
        float
            Estimated box-counting dimension
        """
        # Reshape to 2D grid if possible, otherwise use 1D
        n = len(x)
        
        # For 1D signal, embed in 2D using delay embedding
        if n < 4:
            return 1.0
        
        # Create delay embedding
        tau = max(1, n // 10)
        m = min(3, n // tau)
        
        if m < 2:
            return 1.0
        
        embedded = np.array([x[i:i+m*tau:tau] for i in range(n - m*tau + 1)])
        
        if len(embedded) == 0:
            return 1.0
        
        # Normalize
        embedded = (embedded - embedded.min()) / (embedded.max() - embedded.min() + 1e-10)
        
        # Box counting
        scales = np.logspace(0, np.log10(min(embedded.shape)), self.n_scales)
        counts = []
        
        for scale in scales:
            if scale < 1:
                scale = 1
            # Discretize
            bins = int(1.0 / scale) + 1
            if bins < 2:
                bins = 2
            
            # Count occupied boxes
            digitized = (embedded * (bins - 1)).astype(int)
            digitized = np.clip(digitized, 0, bins - 1)
            unique_boxes = len(set(map(tuple, digitized)))
            counts.append(unique_boxes)
        
        # Fit log-log plot
        counts = np.array(counts)
        valid = counts > 0
        
        if valid.sum() < 2:
            return 1.0
        
        log_scales = np.log(scales[valid])
        log_counts = np.log(counts[valid])
        
        # Linear regression
        coeffs = np.polyfit(log_scales, log_counts, 1)
        dimension = -coeffs[0]
        
        return max(0.1, min(dimension, len(x)))
    
    def _correlation_dimension(self, x):
        """
        Estimate correlation dimension.
        
        Parameters
        ----------
        x : array-like of shape (n_features,)
            Input sample
            
        Returns
        -------
        float
            Estimated correlation dimension
        """
        n = len(x)
        if n < 4:
            return 1.0
        
        # Delay embedding
        tau = max(1, n // 10)
        m = min(3, n // tau)
        
        if m < 2:
            return 1.0
        
        embedded = np.array([x[i:i+m*tau:tau] for i in range(n - m*tau + 1)])
        
        if len(embedded) < 2:
            return 1.0
        
        # Compute pairwise distances
        distances = pdist(embedded, metric='euclidean')
        
        if len(distances) == 0:
            return 1.0
        
        # Correlation sum at different scales
        scales = np.logspace(np.log10(distances.min() + 1e-10), 
                            np.log10(distances.max() + 1e-10), 
                            self.n_scales)
        
        corr_sums = []
        for r in scales:
            corr_sum = np.sum(distances < r) / len(distances)
            if corr_sum > 0:
                corr_sums.append(corr_sum)
            else:
                corr_sums.append(1e-10)
        
        corr_sums = np.array(corr_sums)
        
        # Fit log-log
        log_scales = np.log(scales)
        log_corr = np.log(corr_sums)
        
        valid = np.isfinite(log_scales) & np.isfinite(log_corr)
        
        if valid.sum() < 2:
            return 1.0
        
        coeffs = np.polyfit(log_scales[valid], log_corr[valid], 1)
        dimension = coeffs[0]
        
        return max(0.1, min(dimension, len(x)))
    
    def _compute_fractal_dimension(self, x):
        """
        Compute combined fractal dimension.
        
        Parameters
        ----------
        x : array-like
            Input sample
            
        Returns
        -------
        float
            Combined fractal dimension
        """
        box_dim = self._box_counting_dimension(x)
        corr_dim = self._correlation_dimension(x)
        
        # Average the two estimates
        return (box_dim + corr_dim) / 2.0
    
    def _compute_features(self, X):
        """
        Compute Kolmogorov complexity and fractal dimensions for samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns
        -------
        kolmogorov : ndarray of shape (n_samples,)
            Kolmogorov complexity estimates
        fractal : ndarray of shape (n_samples,)
            Fractal dimension estimates
        """
        n_samples = X.shape[0]
        kolmogorov = np.zeros(n_samples)
        fractal = np.zeros(n_samples)
        
        for i in range(n_samples):
            kolmogorov[i] = self._kolmogorov_complexity(X[i])
            fractal[i] = self._compute_fractal_dimension(X[i])
        
        return kolmogorov, fractal
    
    def _weighted_distance(self, x1, x2, k1, k2, f1, f2):
        """
        Compute weighted distance between two samples.
        
        Parameters
        ----------
        x1, x2 : array-like
            Input samples
        k1, k2 : float
            Kolmogorov complexities
        f1, f2 : float
            Fractal dimensions
            
        Returns
        -------
        float
            Weighted distance
        """
        # Euclidean distance
        euclidean_dist = np.linalg.norm(x1 - x2)
        
        # Kolmogorov distance (normalized compression distance)
        concat = np.concatenate([x1, x2])
        k_concat = self._kolmogorov_complexity(concat)
        ncd = (k_concat - min(k1, k2)) / max(k1, k2)
        
        # Fractal dimension difference
        fractal_dist = abs(f1 - f2)
        
        # Algorithmic information weight
        # Higher complexity = more information = higher weight
        info_weight = (k1 + k2) / 2.0
        fractal_weight = (f1 + f2) / 2.0
        
        if self.metric == 'kolmogorov':
            return ncd
        elif self.metric == 'fractal':
            return euclidean_dist * fractal_weight
        else:  # weighted
            # Combine distances weighted by algorithmic information
            weighted_dist = (self.alpha * ncd * info_weight + 
                           (1 - self.alpha) * euclidean_dist * fractal_weight)
            return weighted_dist
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Check that X and y have correct shape
        X_train, y_train = check_X_y(X_train, y_train)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y_train)
        
        # Store training data
        self.X_train_ = X_train
        self.y_train_ = y_train
        
        # Compute Kolmogorov complexity and fractal dimensions
        self.kolmogorov_complexity_, self.fractal_dims_ = self._compute_features(X_train)
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        # Check if fit has been called
        check_is_fitted(self, ['X_train_', 'y_train_', 
                               'kolmogorov_complexity_', 'fractal_dims_'])
        
        # Input validation
        X_test = check_array(X_test)
        
        # Compute features for test samples
        test_kolmogorov, test_fractal = self._compute_features(X_test)
        
        # Predict using weighted k-NN (k=1 for simplicity)
        n_test = X_test.shape[0]
        n_train = self.X_train_.shape[0]
        
        predictions = np.zeros(n_test, dtype=self.y_train_.dtype)
        
        for i in range(n_test):
            # Compute distances to all training samples
            distances = np.zeros(n_train)
            
            for j in range(n_train):
                distances[j] = self._weighted_distance(
                    X_test[i], self.X_train_[j],
                    test_kolmogorov[i], self.kolmogorov_complexity_[j],
                    test_fractal[i], self.fractal_dims_[j]
                )
            
            # Find nearest neighbor
            nearest_idx = np.argmin(distances)
            predictions[i] = self.y_train_[nearest_idx]
        
        return predictions
    
    def predict_proba(self, X_test, k=5):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples
        k : int, default=5
            Number of neighbors to consider
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self, ['X_train_', 'y_train_', 
                               'kolmogorov_complexity_', 'fractal_dims_'])
        
        X_test = check_array(X_test)
        
        test_kolmogorov, test_fractal = self._compute_features(X_test)
        
        n_test = X_test.shape[0]
        n_train = self.X_train_.shape[0]
        n_classes = len(self.classes_)
        
        proba = np.zeros((n_test, n_classes))
        
        for i in range(n_test):
            distances = np.zeros(n_train)
            
            for j in range(n_train):
                distances[j] = self._weighted_distance(
                    X_test[i], self.X_train_[j],
                    test_kolmogorov[i], self.kolmogorov_complexity_[j],
                    test_fractal[i], self.fractal_dims_[j]
                )
            
            # Find k nearest neighbors
            k_nearest = np.argsort(distances)[:k]
            k_labels = self.y_train_[k_nearest]
            
            # Compute probabilities
            for class_idx, class_label in enumerate(self.classes_):
                proba[i, class_idx] = np.sum(k_labels == class_label) / k
        
        return proba