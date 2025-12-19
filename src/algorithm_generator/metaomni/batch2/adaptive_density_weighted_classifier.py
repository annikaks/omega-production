import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


class AdaptiveDensityWeightedClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Density-Based Weighted Classifier.
    
    This classifier uses adaptive density-based weights that account for local
    sample concentration in feature space, replacing fixed exponential reweighting.
    Samples in dense regions receive lower weights while samples in sparse regions
    receive higher weights, improving classification in imbalanced feature spaces.
    
    Parameters
    ----------
    n_neighbors : int, default=10
        Number of neighbors to use for density estimation.
    
    density_metric : str, default='euclidean'
        Distance metric for density computation.
    
    bandwidth : float or 'auto', default='auto'
        Bandwidth for density estimation. If 'auto', uses median distance.
    
    alpha : float, default=1.0
        Exponent for density-based weight transformation.
        Higher values increase the contrast between dense and sparse regions.
    
    distance_metric : str, default='euclidean'
        Distance metric for prediction.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes seen during fit.
    
    X_ : ndarray of shape (n_samples, n_features)
        The training input samples.
    
    y_ : ndarray of shape (n_samples,)
        The training labels.
    
    weights_ : ndarray of shape (n_samples,)
        Adaptive density-based weights for each training sample.
    
    densities_ : ndarray of shape (n_samples,)
        Local density estimates for each training sample.
    """
    
    def __init__(self, n_neighbors=10, density_metric='euclidean', 
                 bandwidth='auto', alpha=1.0, distance_metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.density_metric = density_metric
        self.bandwidth = bandwidth
        self.alpha = alpha
        self.distance_metric = distance_metric
    
    def _compute_local_density(self, X):
        """
        Compute local density for each sample based on k-nearest neighbors.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.
        
        Returns
        -------
        densities : ndarray of shape (n_samples,)
            Local density estimates.
        """
        n_samples = X.shape[0]
        k = min(self.n_neighbors, n_samples - 1)
        
        # Fit nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric=self.density_metric)
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Exclude self (first neighbor)
        distances = distances[:, 1:]
        
        # Compute bandwidth
        if self.bandwidth == 'auto':
            bandwidth = np.median(distances[:, -1])
            if bandwidth == 0:
                bandwidth = 1.0
        else:
            bandwidth = self.bandwidth
        
        # Compute density as inverse of average distance to k-nearest neighbors
        # Add small epsilon to avoid division by zero
        avg_distances = np.mean(distances, axis=1)
        densities = 1.0 / (avg_distances + 1e-10)
        
        return densities
    
    def _compute_adaptive_weights(self, densities):
        """
        Compute adaptive weights from density estimates.
        
        Samples in dense regions get lower weights, samples in sparse regions
        get higher weights.
        
        Parameters
        ----------
        densities : ndarray of shape (n_samples,)
            Local density estimates.
        
        Returns
        -------
        weights : ndarray of shape (n_samples,)
            Adaptive weights.
        """
        # Normalize densities
        densities_normalized = densities / (np.max(densities) + 1e-10)
        
        # Inverse density weighting with alpha exponent
        # Higher density -> lower weight
        weights = (1.0 / (densities_normalized + 1e-10)) ** self.alpha
        
        # Normalize weights to sum to number of samples
        weights = weights / np.mean(weights)
        
        return weights
    
    def fit(self, X_train, y_train):
        """
        Fit the adaptive density-weighted classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        
        y_train : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Check that X and y have correct shape
        X_train, y_train = check_X_y(X_train, y_train)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y_train)
        
        # Store training data
        self.X_ = X_train
        self.y_ = y_train
        
        # Compute local densities
        self.densities_ = self._compute_local_density(X_train)
        
        # Compute adaptive weights
        self.weights_ = self._compute_adaptive_weights(self.densities_)
        
        return self
    
    def _weighted_vote(self, distances, neighbor_indices):
        """
        Perform weighted voting based on distances and adaptive weights.
        
        Parameters
        ----------
        distances : ndarray of shape (n_queries, n_neighbors)
            Distances to neighbors.
        
        neighbor_indices : ndarray of shape (n_queries, n_neighbors)
            Indices of neighbors.
        
        Returns
        -------
        predictions : ndarray of shape (n_queries,)
            Predicted class labels.
        """
        n_queries = distances.shape[0]
        predictions = np.zeros(n_queries, dtype=self.classes_.dtype)
        
        for i in range(n_queries):
            # Get neighbor labels and weights
            neighbor_labels = self.y_[neighbor_indices[i]]
            neighbor_weights = self.weights_[neighbor_indices[i]]
            
            # Weight by inverse distance and adaptive density weights
            # Add small epsilon to avoid division by zero
            distance_weights = 1.0 / (distances[i] + 1e-10)
            combined_weights = distance_weights * neighbor_weights
            
            # Weighted voting
            class_votes = {}
            for label, weight in zip(neighbor_labels, combined_weights):
                if label not in class_votes:
                    class_votes[label] = 0
                class_votes[label] += weight
            
            # Predict class with highest weighted vote
            predictions[i] = max(class_votes.items(), key=lambda x: x[1])[0]
        
        return predictions
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_queries, n_features)
            Test samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_queries,)
            Predicted class labels.
        """
        # Check if fit has been called
        check_is_fitted(self, ['X_', 'y_', 'weights_'])
        
        # Input validation
        X_test = check_array(X_test)
        
        # Compute distances to all training samples
        distances = cdist(X_test, self.X_, metric=self.distance_metric)
        
        # Find k nearest neighbors
        k = min(self.n_neighbors, self.X_.shape[0])
        neighbor_indices = np.argsort(distances, axis=1)[:, :k]
        neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
        
        # Perform weighted voting
        predictions = self._weighted_vote(neighbor_distances, neighbor_indices)
        
        return predictions
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_queries, n_features)
            Test samples.
        
        Returns
        -------
        proba : ndarray of shape (n_queries, n_classes)
            Class probabilities.
        """
        # Check if fit has been called
        check_is_fitted(self, ['X_', 'y_', 'weights_'])
        
        # Input validation
        X_test = check_array(X_test)
        
        # Compute distances to all training samples
        distances = cdist(X_test, self.X_, metric=self.distance_metric)
        
        # Find k nearest neighbors
        k = min(self.n_neighbors, self.X_.shape[0])
        neighbor_indices = np.argsort(distances, axis=1)[:, :k]
        neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
        
        n_queries = X_test.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_queries, n_classes))
        
        for i in range(n_queries):
            # Get neighbor labels and weights
            neighbor_labels = self.y_[neighbor_indices[i]]
            neighbor_weights = self.weights_[neighbor_indices[i]]
            
            # Weight by inverse distance and adaptive density weights
            distance_weights = 1.0 / (neighbor_distances[i] + 1e-10)
            combined_weights = distance_weights * neighbor_weights
            
            # Compute weighted votes for each class
            for j, cls in enumerate(self.classes_):
                mask = neighbor_labels == cls
                proba[i, j] = np.sum(combined_weights[mask])
            
            # Normalize to probabilities
            proba[i] /= (np.sum(proba[i]) + 1e-10)
        
        return proba