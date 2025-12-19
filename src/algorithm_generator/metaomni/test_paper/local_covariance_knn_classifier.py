import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.linalg import pinvh
from scipy.stats import multivariate_normal


class LocalCovarianceKNNClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses locally adaptive covariance matrices estimated from
    k-nearest neighbors in feature space for classification.
    
    This classifier computes class-conditional probability densities using
    Gaussian distributions with covariance matrices estimated locally from
    k-nearest neighbors, rather than using a single global covariance matrix.
    
    Parameters
    ----------
    n_neighbors : int, default=15
        Number of neighbors to use for local covariance estimation.
    
    reg_covar : float, default=1e-6
        Regularization parameter added to the diagonal of covariance matrices
        to ensure positive definiteness.
    
    metric : str, default='euclidean'
        Distance metric to use for finding nearest neighbors.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    
    X_ : ndarray of shape (n_samples, n_features)
        Training data.
    
    y_ : ndarray of shape (n_samples,)
        Training labels.
    
    class_priors_ : dict
        Prior probabilities for each class.
    """
    
    def __init__(self, n_neighbors=15, reg_covar=1e-6, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.reg_covar = reg_covar
        self.metric = metric
    
    def fit(self, X_train, y_train):
        """
        Fit the Local Covariance KNN classifier.
        
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
        
        # Store training data
        self.X_ = X_train
        self.y_ = y_train
        
        # Compute class priors
        self.class_priors_ = {}
        for c in self.classes_:
            self.class_priors_[c] = np.mean(y_train == c)
        
        # Fit nearest neighbors models for each class
        self.nn_models_ = {}
        self.class_data_ = {}
        
        for c in self.classes_:
            class_mask = y_train == c
            X_class = X_train[class_mask]
            self.class_data_[c] = X_class
            
            # Only fit if we have enough samples
            if len(X_class) > 1:
                n_neighbors = min(self.n_neighbors, len(X_class))
                nn_model = NearestNeighbors(
                    n_neighbors=n_neighbors,
                    metric=self.metric
                )
                nn_model.fit(X_class)
                self.nn_models_[c] = nn_model
        
        return self
    
    def _estimate_local_covariance(self, x, X_neighbors):
        """
        Estimate local covariance matrix from neighbors.
        
        Parameters
        ----------
        x : array-like of shape (n_features,)
            Query point.
        
        X_neighbors : array-like of shape (n_neighbors, n_features)
            Neighbor points.
        
        Returns
        -------
        cov : ndarray of shape (n_features, n_features)
            Local covariance matrix.
        
        mean : ndarray of shape (n_features,)
            Local mean vector.
        """
        # Compute local mean
        mean = np.mean(X_neighbors, axis=0)
        
        # Compute local covariance
        centered = X_neighbors - mean
        cov = np.dot(centered.T, centered) / len(X_neighbors)
        
        # Add regularization to ensure positive definiteness
        cov += self.reg_covar * np.eye(cov.shape[0])
        
        return cov, mean
    
    def _compute_log_likelihood(self, x, class_label):
        """
        Compute log-likelihood of x belonging to class_label using local covariance.
        
        Parameters
        ----------
        x : array-like of shape (n_features,)
            Query point.
        
        class_label : int or str
            Class label.
        
        Returns
        -------
        log_likelihood : float
            Log-likelihood value.
        """
        X_class = self.class_data_[class_label]
        
        # Handle edge cases
        if len(X_class) == 0:
            return -np.inf
        
        if len(X_class) == 1:
            # Use a simple distance-based measure
            dist = np.linalg.norm(x - X_class[0])
            return -dist
        
        # Find k-nearest neighbors in this class
        nn_model = self.nn_models_[class_label]
        n_neighbors = min(self.n_neighbors, len(X_class))
        
        distances, indices = nn_model.kneighbors(
            x.reshape(1, -1),
            n_neighbors=n_neighbors
        )
        
        # Get neighbor points
        X_neighbors = X_class[indices[0]]
        
        # Estimate local covariance and mean
        cov, mean = self._estimate_local_covariance(x, X_neighbors)
        
        # Compute log-likelihood using multivariate normal
        try:
            # Use pseudo-inverse for numerical stability
            cov_inv = pinvh(cov)
            
            # Compute log determinant
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                logdet = 0  # Fallback for numerical issues
            
            # Compute Mahalanobis distance
            diff = x - mean
            mahal_dist = np.dot(diff, np.dot(cov_inv, diff))
            
            # Compute log-likelihood
            n_features = len(x)
            log_likelihood = -0.5 * (n_features * np.log(2 * np.pi) + logdet + mahal_dist)
            
        except np.linalg.LinAlgError:
            # Fallback to distance-based measure
            log_likelihood = -np.min(distances[0])
        
        return log_likelihood
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        n_classes = len(self.classes_)
        
        # Compute log posterior for each class
        log_posteriors = np.zeros((n_samples, n_classes))
        
        for i, x in enumerate(X_test):
            for j, c in enumerate(self.classes_):
                # Log posterior = log likelihood + log prior
                log_likelihood = self._compute_log_likelihood(x, c)
                log_prior = np.log(self.class_priors_[c])
                log_posteriors[i, j] = log_likelihood + log_prior
        
        # Convert log posteriors to probabilities using log-sum-exp trick
        log_posteriors_max = np.max(log_posteriors, axis=1, keepdims=True)
        posteriors = np.exp(log_posteriors - log_posteriors_max)
        posteriors /= np.sum(posteriors, axis=1, keepdims=True)
        
        return posteriors
    
    def predict(self, X_test):
        """
        Predict class labels for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]