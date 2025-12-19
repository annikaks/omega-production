import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.linalg import sqrtm, inv, logm, expm
from scipy.spatial.distance import mahalanobis


class GeodesicCovarianceEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier using geodesic interpolation on the manifold of 
    positive definite matrices (covariance matrices).
    
    This classifier uses the Riemannian geometry of the manifold of symmetric
    positive definite (SPD) matrices to perform geodesic averaging of class
    covariances instead of Euclidean averaging.
    
    Parameters
    ----------
    n_estimators : int, default=5
        Number of bootstrap samples to create for ensemble.
    
    reg_param : float, default=1e-6
        Regularization parameter added to covariance matrices for numerical stability.
    
    max_iter : int, default=50
        Maximum iterations for geodesic mean computation.
    
    tol : float, default=1e-6
        Convergence tolerance for geodesic mean computation.
    
    random_state : int, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    
    geodesic_means_ : dict
        Dictionary mapping class labels to their geodesic mean covariance matrices.
    
    class_priors_ : dict
        Dictionary mapping class labels to their prior probabilities.
    
    class_means_ : dict
        Dictionary mapping class labels to their feature means.
    """
    
    def __init__(self, n_estimators=5, reg_param=1e-6, max_iter=50, 
                 tol=1e-6, random_state=None):
        self.n_estimators = n_estimators
        self.reg_param = reg_param
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
    
    def _make_spd(self, matrix):
        """Ensure matrix is symmetric positive definite."""
        matrix = (matrix + matrix.T) / 2
        matrix += self.reg_param * np.eye(matrix.shape[0])
        return matrix
    
    def _log_map(self, P, X):
        """Logarithmic map at point P for matrix X on SPD manifold."""
        P_sqrt = sqrtm(P)
        P_inv_sqrt = inv(P_sqrt)
        return P_sqrt @ logm(P_inv_sqrt @ X @ P_inv_sqrt) @ P_sqrt
    
    def _exp_map(self, P, V):
        """Exponential map at point P for tangent vector V on SPD manifold."""
        P_sqrt = sqrtm(P)
        P_inv_sqrt = inv(P_sqrt)
        return P_sqrt @ expm(P_inv_sqrt @ V @ P_inv_sqrt) @ P_sqrt
    
    def _geodesic_mean(self, matrices):
        """
        Compute the Riemannian (geodesic) mean of SPD matrices.
        Uses the Karcher flow algorithm.
        """
        if len(matrices) == 1:
            return matrices[0]
        
        # Initialize with Euclidean mean
        G = np.mean(matrices, axis=0)
        G = self._make_spd(G)
        
        for iteration in range(self.max_iter):
            # Compute logarithmic maps
            log_maps = [self._log_map(G, M) for M in matrices]
            
            # Compute mean of tangent vectors
            mean_log = np.mean(log_maps, axis=0)
            
            # Check convergence
            if np.linalg.norm(mean_log) < self.tol:
                break
            
            # Update via exponential map
            G = self._exp_map(G, mean_log)
            G = self._make_spd(G)
        
        return G
    
    def _bootstrap_sample(self, X, y, rng):
        """Create a bootstrap sample."""
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def _compute_covariance(self, X, mean):
        """Compute covariance matrix for data X with given mean."""
        X_centered = X - mean
        cov = (X_centered.T @ X_centered) / max(X.shape[0] - 1, 1)
        return self._make_spd(cov)
    
    def fit(self, X_train, y_train):
        """
        Fit the geodesic covariance ensemble classifier.
        
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
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Store covariance matrices for each bootstrap sample and class
        class_covariances = {c: [] for c in self.classes_}
        class_means_list = {c: [] for c in self.classes_}
        
        # Create bootstrap samples and compute covariances
        for _ in range(self.n_estimators):
            X_boot, y_boot = self._bootstrap_sample(X_train, y_train, rng)
            
            for c in self.classes_:
                X_class = X_boot[y_boot == c]
                
                if len(X_class) > 1:
                    mean_c = np.mean(X_class, axis=0)
                    cov_c = self._compute_covariance(X_class, mean_c)
                    
                    class_covariances[c].append(cov_c)
                    class_means_list[c].append(mean_c)
        
        # Compute geodesic means of covariances for each class
        self.geodesic_means_ = {}
        self.class_means_ = {}
        
        for c in self.classes_:
            if len(class_covariances[c]) > 0:
                self.geodesic_means_[c] = self._geodesic_mean(
                    class_covariances[c]
                )
                self.class_means_[c] = np.mean(class_means_list[c], axis=0)
        
        # Compute class priors
        self.class_priors_ = {}
        for c in self.classes_:
            self.class_priors_[c] = np.sum(y_train == c) / len(y_train)
        
        return self
    
    def _mahalanobis_distance(self, x, mean, cov):
        """Compute Mahalanobis distance."""
        try:
            cov_inv = inv(cov)
            diff = x - mean
            return diff @ cov_inv @ diff
        except np.linalg.LinAlgError:
            # Fallback to Euclidean distance if inversion fails
            return np.sum((x - mean) ** 2)
    
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
        
        # Compute log-likelihoods using Mahalanobis distance
        log_proba = np.zeros((n_samples, n_classes))
        
        for idx, c in enumerate(self.classes_):
            mean_c = self.class_means_[c]
            cov_c = self.geodesic_means_[c]
            
            # Log determinant term
            sign, logdet = np.linalg.slogdet(cov_c)
            
            for i in range(n_samples):
                # Mahalanobis distance
                dist = self._mahalanobis_distance(X_test[i], mean_c, cov_c)
                
                # Log probability (negative log of Gaussian PDF)
                log_proba[i, idx] = (
                    np.log(self.class_priors_[c]) 
                    - 0.5 * logdet 
                    - 0.5 * dist
                )
        
        # Convert log probabilities to probabilities
        # Subtract max for numerical stability
        log_proba -= np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba)
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
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]