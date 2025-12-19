import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


class MultiScaleKernelEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Multi-Scale Kernel Ensemble Classifier using weighted voting based on local data density.
    
    This classifier uses multiple RBF kernels at different bandwidth scales and combines
    their predictions through weighted voting, where weights are determined by local data density.
    
    Parameters
    ----------
    bandwidth_scales : array-like, default=[0.1, 0.5, 1.0, 2.0, 5.0]
        Scales for kernel bandwidths (gamma = 1 / (2 * scale^2))
    
    n_neighbors : int, default=10
        Number of neighbors to consider for local density estimation
    
    density_power : float, default=1.0
        Power to raise density weights to (higher values give more weight to dense regions)
    
    kernel : str, default='rbf'
        Kernel type to use ('rbf' or 'laplacian')
    """
    
    def __init__(self, bandwidth_scales=None, n_neighbors=10, density_power=1.0, kernel='rbf'):
        if bandwidth_scales is None:
            bandwidth_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
        self.bandwidth_scales = bandwidth_scales
        self.n_neighbors = n_neighbors
        self.density_power = density_power
        self.kernel = kernel
    
    def _compute_kernel(self, X1, X2, gamma):
        """Compute kernel matrix between X1 and X2."""
        distances = cdist(X1, X2, metric='euclidean')
        
        if self.kernel == 'rbf':
            return np.exp(-gamma * distances ** 2)
        elif self.kernel == 'laplacian':
            return np.exp(-gamma * distances)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _estimate_local_density(self, X, X_ref):
        """
        Estimate local data density at points X based on reference data X_ref.
        
        Uses k-nearest neighbors distance as a proxy for density.
        """
        n_neighbors = min(self.n_neighbors, len(X_ref) - 1)
        if n_neighbors < 1:
            return np.ones(len(X))
        
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
        nbrs.fit(X_ref)
        distances, _ = nbrs.kneighbors(X)
        
        # Density is inversely proportional to average distance to neighbors
        # Add small epsilon to avoid division by zero
        avg_distances = np.mean(distances, axis=1)
        density = 1.0 / (avg_distances + 1e-10)
        
        # Normalize densities
        density = density / (np.sum(density) + 1e-10)
        
        return density ** self.density_power
    
    def _predict_single_kernel(self, K_test_train, y_train, density_weights):
        """
        Predict using a single kernel matrix with density-weighted voting.
        
        Parameters
        ----------
        K_test_train : array-like, shape (n_test, n_train)
            Kernel matrix between test and training samples
        y_train : array-like, shape (n_train,)
            Training labels
        density_weights : array-like, shape (n_test,)
            Density weights for test samples
        
        Returns
        -------
        predictions : array-like, shape (n_test,)
            Predicted class labels
        vote_scores : array-like, shape (n_test, n_classes)
            Vote scores for each class
        """
        n_test = K_test_train.shape[0]
        n_classes = len(self.classes_)
        vote_scores = np.zeros((n_test, n_classes))
        
        # For each test sample, compute weighted votes
        for i in range(n_test):
            kernel_weights = K_test_train[i, :]
            
            # Weight by both kernel similarity and density
            combined_weights = kernel_weights * density_weights[i]
            
            # Accumulate votes for each class
            for class_idx, class_label in enumerate(self.classes_):
                class_mask = (y_train == class_label)
                vote_scores[i, class_idx] = np.sum(combined_weights[class_mask])
        
        predictions = self.classes_[np.argmax(vote_scores, axis=1)]
        return predictions, vote_scores
    
    def fit(self, X_train, y_train):
        """
        Fit the Multi-Scale Kernel Ensemble Classifier.
        
        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
            Training data
        y_train : array-like, shape (n_samples,)
            Target values
        
        Returns
        -------
        self : object
            Returns self
        """
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        
        # Store classes and training data
        self.classes_ = unique_labels(y_train)
        self.X_train_ = X_train
        self.y_train_ = y_train
        
        # Compute gammas from bandwidth scales
        self.gammas_ = [1.0 / (2 * scale ** 2) for scale in self.bandwidth_scales]
        
        # Estimate training data density (for normalization purposes)
        self.train_density_ = self._estimate_local_density(X_train, X_train)
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test data
        
        Returns
        -------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels
        """
        # Check if fit has been called
        check_is_fitted(self, ['X_train_', 'y_train_', 'classes_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        n_test = X_test.shape[0]
        n_classes = len(self.classes_)
        n_kernels = len(self.gammas_)
        
        # Estimate local density for test points
        test_density = self._estimate_local_density(X_test, self.X_train_)
        
        # Accumulate votes from all kernels
        ensemble_votes = np.zeros((n_test, n_classes))
        
        for gamma in self.gammas_:
            # Compute kernel matrix between test and training data
            K_test_train = self._compute_kernel(X_test, self.X_train_, gamma)
            
            # Get predictions and vote scores from this kernel
            _, vote_scores = self._predict_single_kernel(
                K_test_train, self.y_train_, test_density
            )
            
            # Accumulate votes
            ensemble_votes += vote_scores
        
        # Average votes across kernels
        ensemble_votes /= n_kernels
        
        # Final prediction based on majority voting
        y_pred = self.classes_[np.argmax(ensemble_votes, axis=1)]
        
        return y_pred
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like, shape (n_samples, n_features)
            Test data
        
        Returns
        -------
        proba : array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        # Check if fit has been called
        check_is_fitted(self, ['X_train_', 'y_train_', 'classes_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        n_test = X_test.shape[0]
        n_classes = len(self.classes_)
        n_kernels = len(self.gammas_)
        
        # Estimate local density for test points
        test_density = self._estimate_local_density(X_test, self.X_train_)
        
        # Accumulate votes from all kernels
        ensemble_votes = np.zeros((n_test, n_classes))
        
        for gamma in self.gammas_:
            # Compute kernel matrix between test and training data
            K_test_train = self._compute_kernel(X_test, self.X_train_, gamma)
            
            # Get vote scores from this kernel
            _, vote_scores = self._predict_single_kernel(
                K_test_train, self.y_train_, test_density
            )
            
            # Accumulate votes
            ensemble_votes += vote_scores
        
        # Average votes across kernels
        ensemble_votes /= n_kernels
        
        # Normalize to get probabilities
        row_sums = np.sum(ensemble_votes, axis=1, keepdims=True)
        proba = ensemble_votes / (row_sums + 1e-10)
        
        return proba