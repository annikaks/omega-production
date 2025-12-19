import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


class AdaptiveKNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive k-Nearest Neighbors Classifier with local kernel density estimation.
    
    Uses fewer neighbors in dense regions and more neighbors in sparse regions
    based on local density estimation.
    
    Parameters
    ----------
    k_min : int, default=3
        Minimum number of neighbors to consider.
    
    k_max : int, default=50
        Maximum number of neighbors to consider.
    
    density_neighbors : int, default=10
        Number of neighbors to use for local density estimation.
    
    metric : str, default='euclidean'
        Distance metric to use for nearest neighbor search.
    
    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        Training data.
    
    y_ : ndarray of shape (n_samples,)
        Training labels.
    
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    """
    
    def __init__(self, k_min=3, k_max=50, density_neighbors=10, metric='euclidean'):
        self.k_min = k_min
        self.k_max = k_max
        self.density_neighbors = density_neighbors
        self.metric = metric
    
    def _estimate_local_density(self, X, reference_points):
        """
        Estimate local density at reference points using kernel density estimation.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data points.
        
        reference_points : ndarray of shape (n_reference, n_features)
            Points at which to estimate density.
        
        Returns
        -------
        densities : ndarray of shape (n_reference,)
            Estimated densities at reference points.
        """
        # Use k-nearest neighbors distance as density proxy
        nbrs = NearestNeighbors(
            n_neighbors=min(self.density_neighbors, len(X)),
            metric=self.metric
        )
        nbrs.fit(X)
        
        distances, _ = nbrs.kneighbors(reference_points)
        
        # Density is inversely proportional to average distance to k nearest neighbors
        # Add small epsilon to avoid division by zero
        avg_distances = np.mean(distances, axis=1)
        densities = 1.0 / (avg_distances + 1e-10)
        
        return densities
    
    def _compute_adaptive_k(self, densities):
        """
        Compute adaptive k values based on local densities.
        
        Parameters
        ----------
        densities : ndarray of shape (n_samples,)
            Local density estimates.
        
        Returns
        -------
        k_values : ndarray of shape (n_samples,)
            Adaptive k values for each sample.
        """
        # Normalize densities to [0, 1]
        min_density = np.min(densities)
        max_density = np.max(densities)
        
        if max_density - min_density < 1e-10:
            # All densities are the same
            normalized_densities = np.ones_like(densities) * 0.5
        else:
            normalized_densities = (densities - min_density) / (max_density - min_density)
        
        # High density -> low k, Low density -> high k
        # Use inverse relationship
        k_values = self.k_max - (normalized_densities * (self.k_max - self.k_min))
        k_values = np.clip(k_values, self.k_min, self.k_max).astype(int)
        
        return k_values
    
    def fit(self, X_train, y_train):
        """
        Fit the adaptive k-NN classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        
        y_train : array-like of shape (n_samples,)
            Target labels.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        
        # Store classes
        self.classes_ = unique_labels(y_train)
        
        # Store training data
        self.X_ = X_train
        self.y_ = y_train
        
        # Pre-compute densities for training data
        self.train_densities_ = self._estimate_local_density(self.X_, self.X_)
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for test data.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fit has been called
        check_is_fitted(self, ['X_', 'y_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Estimate local densities at test points
        test_densities = self._estimate_local_density(self.X_, X_test)
        
        # Compute adaptive k values for test points
        k_values = self._compute_adaptive_k(test_densities)
        
        # Predict for each test sample
        predictions = np.zeros(len(X_test), dtype=self.y_.dtype)
        
        # Fit nearest neighbors with maximum k
        max_k_needed = min(self.k_max, len(self.X_))
        nbrs = NearestNeighbors(n_neighbors=max_k_needed, metric=self.metric)
        nbrs.fit(self.X_)
        
        # Get all neighbors at once
        distances, indices = nbrs.kneighbors(X_test)
        
        for i in range(len(X_test)):
            # Use adaptive k for this sample
            k_i = min(k_values[i], max_k_needed)
            
            # Get k nearest neighbors
            neighbor_indices = indices[i, :k_i]
            neighbor_labels = self.y_[neighbor_indices]
            
            # Majority voting
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            predictions[i] = unique[np.argmax(counts)]
        
        return predictions
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for test data.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        # Check if fit has been called
        check_is_fitted(self, ['X_', 'y_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Estimate local densities at test points
        test_densities = self._estimate_local_density(self.X_, X_test)
        
        # Compute adaptive k values for test points
        k_values = self._compute_adaptive_k(test_densities)
        
        # Initialize probability matrix
        proba = np.zeros((len(X_test), len(self.classes_)))
        
        # Fit nearest neighbors with maximum k
        max_k_needed = min(self.k_max, len(self.X_))
        nbrs = NearestNeighbors(n_neighbors=max_k_needed, metric=self.metric)
        nbrs.fit(self.X_)
        
        # Get all neighbors at once
        distances, indices = nbrs.kneighbors(X_test)
        
        for i in range(len(X_test)):
            # Use adaptive k for this sample
            k_i = min(k_values[i], max_k_needed)
            
            # Get k nearest neighbors
            neighbor_indices = indices[i, :k_i]
            neighbor_labels = self.y_[neighbor_indices]
            
            # Compute class probabilities
            for j, cls in enumerate(self.classes_):
                proba[i, j] = np.sum(neighbor_labels == cls) / k_i
        
        return proba