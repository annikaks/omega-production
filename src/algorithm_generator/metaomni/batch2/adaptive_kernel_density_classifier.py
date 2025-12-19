import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import cdist


class AdaptiveKernelDensityClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Kernel Density Estimation Classifier with sparsity-based radius adaptation.
    
    This classifier estimates local density non-parametrically using kernel density
    estimation and adaptively sets the bandwidth (radius) per query point based on
    local sparsity.
    
    Parameters
    ----------
    bandwidth : float, default=1.0
        Base bandwidth for kernel density estimation.
    
    kernel : str, default='gaussian'
        Kernel to use. Valid kernels are ['gaussian', 'tophat', 'epanechnikov',
        'exponential', 'linear', 'cosine'].
    
    k_neighbors : int, default=10
        Number of neighbors to consider for local sparsity estimation.
    
    adaptation_factor : float, default=1.5
        Factor to scale bandwidth based on local sparsity. Higher values lead to
        more aggressive adaptation.
    
    metric : str, default='euclidean'
        Distance metric to use for sparsity estimation.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes seen during fit.
    
    X_ : ndarray of shape (n_samples, n_features)
        The training input samples.
    
    y_ : ndarray of shape (n_samples,)
        The training labels.
    
    kde_models_ : dict
        Dictionary mapping each class to its fitted KernelDensity model.
    
    local_densities_ : ndarray of shape (n_samples,)
        Local density estimates for training samples.
    """
    
    def __init__(self, bandwidth=1.0, kernel='gaussian', k_neighbors=10,
                 adaptation_factor=1.5, metric='euclidean'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.k_neighbors = k_neighbors
        self.adaptation_factor = adaptation_factor
        self.metric = metric
    
    def _estimate_local_sparsity(self, X, X_ref=None):
        """
        Estimate local sparsity for each point in X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Query points.
        
        X_ref : ndarray of shape (n_ref_samples, n_features), optional
            Reference points. If None, uses X.
        
        Returns
        -------
        sparsity : ndarray of shape (n_samples,)
            Local sparsity measure (average distance to k nearest neighbors).
        """
        if X_ref is None:
            X_ref = X
        
        # Compute pairwise distances
        distances = cdist(X, X_ref, metric=self.metric)
        
        # For each point, find k+1 nearest neighbors (including itself if X_ref is X)
        k = min(self.k_neighbors + 1, distances.shape[1])
        
        # Sort distances and take k nearest
        sorted_distances = np.sort(distances, axis=1)
        
        # If X_ref is X, exclude the first distance (distance to itself = 0)
        if X_ref is X:
            k_nearest_distances = sorted_distances[:, 1:k]
        else:
            k_nearest_distances = sorted_distances[:, :k-1]
        
        # Compute average distance to k nearest neighbors as sparsity measure
        sparsity = np.mean(k_nearest_distances, axis=1)
        
        return sparsity
    
    def _compute_adaptive_bandwidth(self, sparsity):
        """
        Compute adaptive bandwidth based on local sparsity.
        
        Parameters
        ----------
        sparsity : ndarray of shape (n_samples,)
            Local sparsity measures.
        
        Returns
        -------
        bandwidths : ndarray of shape (n_samples,)
            Adaptive bandwidths for each point.
        """
        # Normalize sparsity to [0, 1] range
        sparsity_normalized = (sparsity - np.min(sparsity)) / (np.ptp(sparsity) + 1e-10)
        
        # Compute adaptive bandwidth: higher sparsity -> larger bandwidth
        bandwidths = self.bandwidth * (1 + self.adaptation_factor * sparsity_normalized)
        
        return bandwidths
    
    def fit(self, X_train, y_train):
        """
        Fit the adaptive kernel density classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        
        y_train : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        
        # Store classes
        self.classes_ = unique_labels(y_train)
        
        # Store training data
        self.X_ = X_train
        self.y_ = y_train
        
        # Estimate local sparsity for training data
        self.local_densities_ = self._estimate_local_sparsity(X_train)
        
        # Fit KDE model for each class
        self.kde_models_ = {}
        for class_label in self.classes_:
            # Get samples for this class
            X_class = X_train[y_train == class_label]
            
            # Fit KDE model with base bandwidth
            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            kde.fit(X_class)
            
            self.kde_models_[class_label] = kde
        
        return self
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities for each sample.
        """
        # Check if fit has been called
        check_is_fitted(self, ['X_', 'y_', 'kde_models_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Estimate local sparsity for test points
        test_sparsity = self._estimate_local_sparsity(X_test, self.X_)
        
        # Compute adaptive bandwidths for test points
        adaptive_bandwidths = self._compute_adaptive_bandwidth(test_sparsity)
        
        # Compute log-likelihood for each class with adaptive bandwidth
        log_proba = np.zeros((X_test.shape[0], len(self.classes_)))
        
        for idx, class_label in enumerate(self.classes_):
            # Get training samples for this class
            X_class = self.X_[self.y_ == class_label]
            
            # For each test point, compute density with adaptive bandwidth
            for i in range(X_test.shape[0]):
                # Create temporary KDE with adaptive bandwidth
                kde_adaptive = KernelDensity(
                    bandwidth=adaptive_bandwidths[i],
                    kernel=self.kernel
                )
                kde_adaptive.fit(X_class)
                
                # Compute log-likelihood
                log_proba[i, idx] = kde_adaptive.score_samples(X_test[i:i+1])[0]
        
        # Convert log-likelihood to probabilities
        # Subtract max for numerical stability
        log_proba_normalized = log_proba - np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba_normalized)
        proba /= np.sum(proba, axis=1, keepdims=True)
        
        return proba
    
    def predict(self, X_test):
        """
        Predict class labels for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Get class probabilities
        proba = self.predict_proba(X_test)
        
        # Return class with highest probability
        return self.classes_[np.argmax(proba, axis=1)]
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        y : array-like of shape (n_samples,)
            True labels for X.
        
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))