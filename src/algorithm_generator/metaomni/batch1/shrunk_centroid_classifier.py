import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class ShrunkCentroidClassifier(BaseEstimator, ClassifierMixin):
    """
    Nearest Shrunken Centroids Classifier with bias-variance decomposition.
    
    This classifier decomposes class centroids into bias and variance components,
    shrinking high-variance dimensions toward the global mean to reduce overfitting.
    
    Parameters
    ----------
    shrinkage_threshold : float, default=0.0
        Threshold for soft-thresholding the standardized centroid differences.
        Higher values lead to more aggressive shrinkage and feature selection.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    centroids_ : ndarray of shape (n_classes, n_features)
        The shrunken centroid for each class.
    global_mean_ : ndarray of shape (n_features,)
        The overall mean across all training samples.
    feature_std_ : ndarray of shape (n_features,)
        The pooled within-class standard deviation for each feature.
    """
    
    def __init__(self, shrinkage_threshold=0.0):
        self.shrinkage_threshold = shrinkage_threshold
    
    def fit(self, X, y):
        """
        Fit the Shrunk Centroid Classifier.
        
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
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # Compute global mean (bias component)
        self.global_mean_ = np.mean(X, axis=0)
        
        # Compute class centroids
        raw_centroids = np.zeros((n_classes, n_features))
        class_counts = np.zeros(n_classes)
        
        for idx, cls in enumerate(self.classes_):
            mask = y == cls
            class_counts[idx] = np.sum(mask)
            raw_centroids[idx] = np.mean(X[mask], axis=0)
        
        # Compute pooled within-class standard deviation (variance component)
        pooled_variance = np.zeros(n_features)
        
        for idx, cls in enumerate(self.classes_):
            mask = y == cls
            X_class = X[mask]
            # Sum of squared deviations from class centroid
            pooled_variance += np.sum((X_class - raw_centroids[idx]) ** 2, axis=0)
        
        # Pooled standard deviation
        self.feature_std_ = np.sqrt(pooled_variance / (n_samples - n_classes))
        
        # Avoid division by zero
        self.feature_std_ = np.where(self.feature_std_ == 0, 1e-10, self.feature_std_)
        
        # Compute standardized centroid differences (mk)
        # mk = (centroid_k - global_mean) / (std * sqrt(1/nk + 1/n))
        self.centroids_ = np.zeros((n_classes, n_features))
        
        for idx, cls in enumerate(self.classes_):
            nk = class_counts[idx]
            # Scaling factor for standard error
            scale_factor = self.feature_std_ * np.sqrt(1.0 / nk + 1.0 / n_samples)
            
            # Standardized difference
            standardized_diff = (raw_centroids[idx] - self.global_mean_) / scale_factor
            
            # Soft thresholding (shrinkage)
            shrunken_diff = self._soft_threshold(standardized_diff, self.shrinkage_threshold)
            
            # Transform back to original scale
            self.centroids_[idx] = self.global_mean_ + shrunken_diff * scale_factor
        
        return self
    
    def _soft_threshold(self, x, threshold):
        """
        Apply soft thresholding operator.
        
        Parameters
        ----------
        x : ndarray
            Input array.
        threshold : float
            Threshold value.
        
        Returns
        -------
        ndarray
            Soft-thresholded array.
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
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
        check_is_fitted(self, ['centroids_', 'global_mean_', 'feature_std_'])
        X = check_array(X)
        
        # Compute squared distances to each centroid
        # Using standardized distances
        distances = np.zeros((X.shape[0], len(self.classes_)))
        
        for idx in range(len(self.classes_)):
            # Standardized squared distance
            diff = X - self.centroids_[idx]
            distances[:, idx] = np.sum((diff / self.feature_std_) ** 2, axis=1)
        
        # Predict class with minimum distance
        y_pred = self.classes_[np.argmin(distances, axis=1)]
        
        return y_pred
    
    def decision_function(self, X):
        """
        Compute the decision function (negative distances to centroids).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        decision : ndarray of shape (n_samples, n_classes)
            Negative squared distances to each class centroid.
        """
        check_is_fitted(self, ['centroids_', 'global_mean_', 'feature_std_'])
        X = check_array(X)
        
        distances = np.zeros((X.shape[0], len(self.classes_)))
        
        for idx in range(len(self.classes_)):
            diff = X - self.centroids_[idx]
            distances[:, idx] = np.sum((diff / self.feature_std_) ** 2, axis=1)
        
        return -distances