import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import cdist


class MixtureOfSubCentroidsClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that replaces single class centroids with a mixture of sub-centroids
    to capture multimodal within-class distributions.
    
    For each class, the training data is clustered into multiple sub-centroids using
    K-Means. During prediction, the distance to all sub-centroids is computed, and
    the class with the nearest sub-centroid is assigned.
    
    Parameters
    ----------
    n_subcentroids : int, default=3
        Number of sub-centroids (clusters) to create for each class.
        
    distance_metric : str, default='euclidean'
        Distance metric to use for nearest sub-centroid computation.
        Options: 'euclidean', 'cosine', 'manhattan', 'chebyshev'
        
    aggregation : str, default='min'
        How to aggregate distances to sub-centroids of a class.
        Options: 'min' (nearest sub-centroid), 'mean' (average distance to all sub-centroids)
        
    random_state : int, default=None
        Random state for K-Means clustering reproducibility.
        
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
        
    subcentroids_ : dict
        Dictionary mapping each class to its sub-centroids array.
        
    subcentroid_weights_ : dict
        Dictionary mapping each class to weights (proportions) of each sub-centroid.
    """
    
    def __init__(self, n_subcentroids=3, distance_metric='euclidean', 
                 aggregation='min', random_state=None):
        self.n_subcentroids = n_subcentroids
        self.distance_metric = distance_metric
        self.aggregation = aggregation
        self.random_state = random_state
        
    def fit(self, X_train, y_train):
        """
        Fit the classifier by computing sub-centroids for each class.
        
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
        
        # Initialize storage for sub-centroids and weights
        self.subcentroids_ = {}
        self.subcentroid_weights_ = {}
        
        # For each class, compute sub-centroids
        for class_label in self.classes_:
            # Get samples belonging to this class
            X_class = X_train[y_train == class_label]
            
            # Determine effective number of sub-centroids
            n_samples_class = X_class.shape[0]
            effective_n_subcentroids = min(self.n_subcentroids, n_samples_class)
            
            if effective_n_subcentroids == 1:
                # If only one sample or one sub-centroid, use the mean
                self.subcentroids_[class_label] = X_class.mean(axis=0, keepdims=True)
                self.subcentroid_weights_[class_label] = np.array([1.0])
            else:
                # Cluster the class samples into sub-centroids
                kmeans = KMeans(
                    n_clusters=effective_n_subcentroids,
                    random_state=self.random_state,
                    n_init=10
                )
                kmeans.fit(X_class)
                
                # Store sub-centroids
                self.subcentroids_[class_label] = kmeans.cluster_centers_
                
                # Compute weights as proportion of samples in each cluster
                labels, counts = np.unique(kmeans.labels_, return_counts=True)
                weights = counts / n_samples_class
                self.subcentroid_weights_[class_label] = weights
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fitted
        check_is_fitted(self, ['subcentroids_', 'classes_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        n_classes = len(self.classes_)
        
        # Compute distances to all sub-centroids for each class
        class_distances = np.zeros((n_samples, n_classes))
        
        for idx, class_label in enumerate(self.classes_):
            subcentroids = self.subcentroids_[class_label]
            weights = self.subcentroid_weights_[class_label]
            
            # Compute distances from test samples to all sub-centroids of this class
            distances = cdist(X_test, subcentroids, metric=self.distance_metric)
            
            # Aggregate distances based on strategy
            if self.aggregation == 'min':
                # Use minimum distance to any sub-centroid
                class_distances[:, idx] = distances.min(axis=1)
            elif self.aggregation == 'mean':
                # Use weighted mean distance to all sub-centroids
                class_distances[:, idx] = np.average(distances, axis=1, weights=weights)
            elif self.aggregation == 'weighted_min':
                # Use minimum weighted distance
                weighted_distances = distances / weights
                class_distances[:, idx] = weighted_distances.min(axis=1)
            else:
                raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        # Predict class with minimum distance
        y_pred_indices = class_distances.argmin(axis=1)
        y_pred = self.classes_[y_pred_indices]
        
        return y_pred
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Probabilities are computed using softmax on negative distances.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        # Check if fitted
        check_is_fitted(self, ['subcentroids_', 'classes_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        n_classes = len(self.classes_)
        
        # Compute distances to all sub-centroids for each class
        class_distances = np.zeros((n_samples, n_classes))
        
        for idx, class_label in enumerate(self.classes_):
            subcentroids = self.subcentroids_[class_label]
            weights = self.subcentroid_weights_[class_label]
            
            # Compute distances from test samples to all sub-centroids of this class
            distances = cdist(X_test, subcentroids, metric=self.distance_metric)
            
            # Aggregate distances based on strategy
            if self.aggregation == 'min':
                class_distances[:, idx] = distances.min(axis=1)
            elif self.aggregation == 'mean':
                class_distances[:, idx] = np.average(distances, axis=1, weights=weights)
            elif self.aggregation == 'weighted_min':
                weighted_distances = distances / weights
                class_distances[:, idx] = weighted_distances.min(axis=1)
        
        # Convert distances to probabilities using softmax on negative distances
        # Add small epsilon to avoid division by zero
        neg_distances = -class_distances
        exp_neg_distances = np.exp(neg_distances - neg_distances.max(axis=1, keepdims=True))
        proba = exp_neg_distances / exp_neg_distances.sum(axis=1, keepdims=True)
        
        return proba