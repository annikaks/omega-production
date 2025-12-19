import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde


class AdaptiveKernelClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Kernel Classifier using locally-adaptive kernels estimated 
    non-parametrically from nearest neighbor distributions in feature space.
    
    Parameters
    ----------
    n_neighbors : int, default=15
        Number of nearest neighbors to use for local kernel estimation.
    
    bandwidth_method : str or float, default='scott'
        Method to compute bandwidth for KDE. Can be 'scott', 'silverman', 
        or a scalar multiplier.
    
    metric : str, default='euclidean'
        Distance metric for nearest neighbor search.
    
    kernel_type : str, default='gaussian'
        Type of kernel to use. Currently supports 'gaussian'.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    
    X_ : ndarray of shape (n_samples, n_features)
        Training data.
    
    y_ : ndarray of shape (n_samples,)
        Training labels.
    """
    
    def __init__(self, n_neighbors=15, bandwidth_method='scott', 
                 metric='euclidean', kernel_type='gaussian'):
        self.n_neighbors = n_neighbors
        self.bandwidth_method = bandwidth_method
        self.metric = metric
        self.kernel_type = kernel_type
    
    def fit(self, X_train, y_train):
        """
        Fit the adaptive kernel classifier.
        
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
        
        # Store classes and training data
        self.classes_ = unique_labels(y_train)
        self.X_ = X_train
        self.y_ = y_train
        
        # Build nearest neighbor index for each class
        self.class_indices_ = {}
        self.nn_models_ = {}
        
        for cls in self.classes_:
            cls_mask = (y_train == cls)
            self.class_indices_[cls] = np.where(cls_mask)[0]
            
            # Build NN model for this class
            X_cls = X_train[cls_mask]
            if len(X_cls) > 0:
                n_neighbors = min(self.n_neighbors, len(X_cls))
                nn_model = NearestNeighbors(
                    n_neighbors=n_neighbors,
                    metric=self.metric
                )
                nn_model.fit(X_cls)
                self.nn_models_[cls] = nn_model
        
        return self
    
    def _estimate_local_bandwidth(self, X_point, X_neighbors):
        """
        Estimate local bandwidth from nearest neighbors.
        
        Parameters
        ----------
        X_point : array-like of shape (n_features,)
            Query point.
        
        X_neighbors : array-like of shape (n_neighbors, n_features)
            Nearest neighbors.
        
        Returns
        -------
        bandwidth : float
            Estimated bandwidth.
        """
        if len(X_neighbors) < 2:
            return 1.0
        
        # Compute local covariance
        cov_matrix = np.cov(X_neighbors.T)
        
        # Handle scalar case
        if cov_matrix.ndim == 0:
            std_dev = np.sqrt(cov_matrix)
        else:
            # Use trace of covariance as a measure of local spread
            std_dev = np.sqrt(np.trace(cov_matrix) / X_neighbors.shape[1])
        
        n = len(X_neighbors)
        d = X_neighbors.shape[1]
        
        # Scott's rule adapted for local estimation
        if self.bandwidth_method == 'scott':
            bandwidth = std_dev * n ** (-1.0 / (d + 4))
        elif self.bandwidth_method == 'silverman':
            bandwidth = std_dev * (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))
        else:
            # Assume it's a scalar multiplier
            bandwidth = std_dev * float(self.bandwidth_method)
        
        return max(bandwidth, 1e-6)  # Avoid zero bandwidth
    
    def _adaptive_kernel_density(self, X_point, X_neighbors, bandwidth):
        """
        Compute adaptive kernel density at a point.
        
        Parameters
        ----------
        X_point : array-like of shape (n_features,)
            Query point.
        
        X_neighbors : array-like of shape (n_neighbors, n_features)
            Nearest neighbors.
        
        bandwidth : float
            Bandwidth parameter.
        
        Returns
        -------
        density : float
            Estimated density.
        """
        if len(X_neighbors) == 0:
            return 0.0
        
        # Compute distances
        distances = cdist([X_point], X_neighbors, metric=self.metric)[0]
        
        # Gaussian kernel
        if self.kernel_type == 'gaussian':
            kernel_values = np.exp(-0.5 * (distances / bandwidth) ** 2)
            kernel_values /= (bandwidth * np.sqrt(2 * np.pi))
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
        
        # Average kernel values
        density = np.mean(kernel_values)
        
        return density
    
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
            Class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        
        for i, x in enumerate(X_test):
            class_densities = []
            
            for cls_idx, cls in enumerate(self.classes_):
                if cls not in self.nn_models_:
                    class_densities.append(0.0)
                    continue
                
                # Get nearest neighbors from this class
                nn_model = self.nn_models_[cls]
                X_cls = self.X_[self.class_indices_[cls]]
                
                # Find nearest neighbors
                distances, indices = nn_model.kneighbors([x])
                X_neighbors = X_cls[indices[0]]
                
                # Estimate local bandwidth
                bandwidth = self._estimate_local_bandwidth(x, X_neighbors)
                
                # Compute adaptive kernel density
                density = self._adaptive_kernel_density(x, X_neighbors, bandwidth)
                
                # Weight by class prior
                class_prior = len(self.class_indices_[cls]) / len(self.y_)
                class_densities.append(density * class_prior)
            
            # Normalize to get probabilities
            total_density = sum(class_densities)
            if total_density > 0:
                proba[i] = np.array(class_densities) / total_density
            else:
                # Uniform distribution if all densities are zero
                proba[i] = np.ones(n_classes) / n_classes
        
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
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]