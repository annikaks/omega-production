import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway


class AdaptiveFeatureWeightingClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Feature Weighting Classifier based on per-class discriminative power.
    
    This classifier computes feature weights based on their discriminative power
    across classes using ANOVA F-statistics, then uses weighted distance metrics
    for classification.
    
    Parameters
    ----------
    weighting_method : str, default='anova'
        Method to compute feature weights. Options: 'anova', 'variance_ratio'
    distance_metric : str, default='euclidean'
        Distance metric for classification. Options: 'euclidean', 'manhattan'
    normalize : bool, default=True
        Whether to normalize features before computing weights
    alpha : float, default=1.0
        Exponent for feature weight scaling (higher values increase weight contrast)
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes seen during fit
    feature_weights_ : ndarray of shape (n_features,)
        Discriminative power scores for each feature
    class_centroids_ : ndarray of shape (n_classes, n_features)
        Weighted centroids for each class
    """
    
    def __init__(self, weighting_method='anova', distance_metric='euclidean',
                 normalize=True, alpha=1.0):
        self.weighting_method = weighting_method
        self.distance_metric = distance_metric
        self.normalize = normalize
        self.alpha = alpha
    
    def _compute_anova_weights(self, X, y):
        """Compute feature weights using ANOVA F-statistic."""
        weights = np.zeros(X.shape[1])
        
        for feature_idx in range(X.shape[1]):
            # Group feature values by class
            groups = [X[y == c, feature_idx] for c in self.classes_]
            
            # Compute F-statistic (handles edge cases)
            if len(groups) > 1 and all(len(g) > 0 for g in groups):
                f_stat, _ = f_oneway(*groups)
                weights[feature_idx] = f_stat if not np.isnan(f_stat) else 0.0
            else:
                weights[feature_idx] = 0.0
        
        return weights
    
    def _compute_variance_ratio_weights(self, X, y):
        """Compute feature weights using between-class to within-class variance ratio."""
        weights = np.zeros(X.shape[1])
        
        for feature_idx in range(X.shape[1]):
            # Between-class variance
            class_means = np.array([X[y == c, feature_idx].mean() 
                                   for c in self.classes_])
            overall_mean = X[:, feature_idx].mean()
            class_counts = np.array([np.sum(y == c) for c in self.classes_])
            
            between_var = np.sum(class_counts * (class_means - overall_mean) ** 2)
            
            # Within-class variance
            within_var = np.sum([np.sum((X[y == c, feature_idx] - class_means[i]) ** 2)
                                for i, c in enumerate(self.classes_)])
            
            # Compute ratio (avoid division by zero)
            if within_var > 1e-10:
                weights[feature_idx] = between_var / within_var
            else:
                weights[feature_idx] = between_var if between_var > 0 else 0.0
        
        return weights
    
    def _normalize_weights(self, weights):
        """Normalize weights to sum to number of features."""
        weights = np.maximum(weights, 0)  # Ensure non-negative
        
        if np.sum(weights) > 1e-10:
            # Normalize so that weights sum to n_features (average weight = 1)
            weights = weights / np.sum(weights) * len(weights)
        else:
            # If all weights are zero, use uniform weights
            weights = np.ones_like(weights)
        
        # Apply alpha scaling
        weights = weights ** self.alpha
        
        return weights
    
    def _compute_weighted_distance(self, X, centroids):
        """Compute weighted distance from samples to class centroids."""
        n_samples = X.shape[0]
        n_classes = centroids.shape[0]
        distances = np.zeros((n_samples, n_classes))
        
        for class_idx in range(n_classes):
            diff = X - centroids[class_idx]
            
            if self.distance_metric == 'euclidean':
                # Weighted Euclidean distance
                distances[:, class_idx] = np.sqrt(
                    np.sum(self.feature_weights_ * diff ** 2, axis=1)
                )
            elif self.distance_metric == 'manhattan':
                # Weighted Manhattan distance
                distances[:, class_idx] = np.sum(
                    self.feature_weights_ * np.abs(diff), axis=1
                )
            else:
                raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distances
    
    def fit(self, X_train, y_train):
        """
        Fit the adaptive feature weighting classifier.
        
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
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        
        # Store classes
        self.classes_ = unique_labels(y_train)
        self.n_features_in_ = X_train.shape[1]
        
        # Normalize features if requested
        if self.normalize:
            self.scaler_ = StandardScaler()
            X_train = self.scaler_.fit_transform(X_train)
        
        # Compute feature weights based on discriminative power
        if self.weighting_method == 'anova':
            raw_weights = self._compute_anova_weights(X_train, y_train)
        elif self.weighting_method == 'variance_ratio':
            raw_weights = self._compute_variance_ratio_weights(X_train, y_train)
        else:
            raise ValueError(f"Unknown weighting method: {self.weighting_method}")
        
        # Normalize weights
        self.feature_weights_ = self._normalize_weights(raw_weights)
        
        # Compute weighted class centroids
        self.class_centroids_ = np.zeros((len(self.classes_), X_train.shape[1]))
        for i, c in enumerate(self.classes_):
            self.class_centroids_[i] = X_train[y_train == c].mean(axis=0)
        
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
        check_is_fitted(self, ['feature_weights_', 'class_centroids_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, "
                           f"got {X_test.shape[1]}")
        
        # Normalize features if requested
        if self.normalize:
            X_test = self.scaler_.transform(X_test)
        
        # Compute weighted distances to class centroids
        distances = self._compute_weighted_distance(X_test, self.class_centroids_)
        
        # Predict class with minimum distance
        predicted_indices = np.argmin(distances, axis=1)
        y_pred = self.classes_[predicted_indices]
        
        return y_pred
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities (based on inverse distances)
        """
        check_is_fitted(self, ['feature_weights_', 'class_centroids_'])
        X_test = check_array(X_test)
        
        if self.normalize:
            X_test = self.scaler_.transform(X_test)
        
        # Compute weighted distances
        distances = self._compute_weighted_distance(X_test, self.class_centroids_)
        
        # Convert distances to probabilities using softmax on negative distances
        # Add small epsilon to avoid division by zero
        neg_distances = -distances
        exp_neg_dist = np.exp(neg_distances - np.max(neg_distances, axis=1, keepdims=True))
        proba = exp_neg_dist / np.sum(exp_neg_dist, axis=1, keepdims=True)
        
        return proba
    
    def get_feature_importance(self):
        """
        Get normalized feature importance scores.
        
        Returns
        -------
        importance : ndarray of shape (n_features,)
            Feature importance scores (normalized to sum to 1)
        """
        check_is_fitted(self, ['feature_weights_'])
        importance = self.feature_weights_ / np.sum(self.feature_weights_)
        return importance