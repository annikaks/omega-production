import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import mahalanobis


class LocalCovarianceMetricClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that learns separate distance metrics per class by estimating
    local covariance structure to capture class-specific feature correlations.
    
    This classifier uses Mahalanobis distance with class-specific covariance
    matrices to measure similarity between test samples and class prototypes.
    
    Parameters
    ----------
    regularization : float, default=1e-6
        Regularization parameter added to diagonal of covariance matrices
        to ensure numerical stability and invertibility.
    
    metric : str, default='mahalanobis'
        Distance metric to use. Currently supports 'mahalanobis'.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    
    class_means_ : dict
        Dictionary mapping each class to its mean feature vector.
    
    class_covariances_ : dict
        Dictionary mapping each class to its covariance matrix.
    
    class_inv_covariances_ : dict
        Dictionary mapping each class to its inverse covariance matrix.
    
    n_features_in_ : int
        Number of features seen during fit.
    """
    
    def __init__(self, regularization=1e-6, metric='mahalanobis'):
        self.regularization = regularization
        self.metric = metric
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier by estimating class-specific means and covariance matrices.
        
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
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y_train)
        self.n_features_in_ = X_train.shape[1]
        
        # Initialize storage for class-specific statistics
        self.class_means_ = {}
        self.class_covariances_ = {}
        self.class_inv_covariances_ = {}
        self.class_priors_ = {}
        
        # Compute class-specific statistics
        for class_label in self.classes_:
            # Get samples belonging to this class
            class_mask = (y_train == class_label)
            X_class = X_train[class_mask]
            
            # Compute class prior
            self.class_priors_[class_label] = np.sum(class_mask) / len(y_train)
            
            # Compute mean
            class_mean = np.mean(X_class, axis=0)
            self.class_means_[class_label] = class_mean
            
            # Compute covariance matrix
            if len(X_class) > 1:
                class_cov = np.cov(X_class, rowvar=False)
                
                # Handle 1D case
                if class_cov.ndim == 0:
                    class_cov = np.array([[class_cov]])
                
                # Add regularization to ensure invertibility
                class_cov += self.regularization * np.eye(self.n_features_in_)
            else:
                # If only one sample, use regularized identity
                class_cov = self.regularization * np.eye(self.n_features_in_)
            
            self.class_covariances_[class_label] = class_cov
            
            # Compute and store inverse covariance matrix
            try:
                self.class_inv_covariances_[class_label] = np.linalg.inv(class_cov)
            except np.linalg.LinAlgError:
                # If inversion fails, add more regularization
                class_cov += 1e-3 * np.eye(self.n_features_in_)
                self.class_inv_covariances_[class_label] = np.linalg.inv(class_cov)
        
        return self
    
    def _mahalanobis_distance(self, x, class_label):
        """
        Compute Mahalanobis distance between sample x and class prototype.
        
        Parameters
        ----------
        x : array-like of shape (n_features,)
            Sample to compute distance for.
        
        class_label : int or str
            Class label to compute distance to.
        
        Returns
        -------
        distance : float
            Mahalanobis distance.
        """
        mean = self.class_means_[class_label]
        inv_cov = self.class_inv_covariances_[class_label]
        
        diff = x - mean
        distance = np.sqrt(np.dot(np.dot(diff, inv_cov), diff))
        
        return distance
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.
        
        Uses negative Mahalanobis distances to compute likelihood scores,
        which are then normalized to probabilities.
        
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
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, "
                           f"got {X_test.shape[1]}")
        
        n_samples = X_test.shape[0]
        n_classes = len(self.classes_)
        
        # Compute distances to each class
        distances = np.zeros((n_samples, n_classes))
        
        for i, class_label in enumerate(self.classes_):
            for j, x in enumerate(X_test):
                distances[j, i] = self._mahalanobis_distance(x, class_label)
        
        # Convert distances to probabilities using negative exponential
        # Smaller distance = higher probability
        log_likelihoods = -0.5 * distances ** 2
        
        # Add log determinant term for proper probability density
        for i, class_label in enumerate(self.classes_):
            sign, logdet = np.linalg.slogdet(self.class_covariances_[class_label])
            log_likelihoods[:, i] -= 0.5 * logdet
            log_likelihoods[:, i] += np.log(self.class_priors_[class_label])
        
        # Normalize to get probabilities (using log-sum-exp trick for stability)
        max_log_likelihood = np.max(log_likelihoods, axis=1, keepdims=True)
        exp_log_likelihoods = np.exp(log_likelihoods - max_log_likelihood)
        proba = exp_log_likelihoods / np.sum(exp_log_likelihoods, axis=1, keepdims=True)
        
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
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, "
                           f"got {X_test.shape[1]}")
        
        # Get probabilities and return class with highest probability
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def decision_function(self, X_test):
        """
        Compute decision function values (negative distances) for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        scores : ndarray of shape (n_samples, n_classes)
            Decision function values (higher is more confident).
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, "
                           f"got {X_test.shape[1]}")
        
        n_samples = X_test.shape[0]
        n_classes = len(self.classes_)
        
        scores = np.zeros((n_samples, n_classes))
        
        for i, class_label in enumerate(self.classes_):
            for j, x in enumerate(X_test):
                # Negative distance (higher is better)
                scores[j, i] = -self._mahalanobis_distance(x, class_label)
        
        return scores