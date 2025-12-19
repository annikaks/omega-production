import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.special import logsumexp


class ConditionalFeaturePriorClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that models class-conditional feature distributions as a linear
    combination with globally shared feature priors.
    
    This approach balances simplicity (shared priors) with expressiveness 
    (class-conditional distributions) by modeling:
    P(x|y) = alpha * P_class(x|y) + (1-alpha) * P_global(x)
    
    Parameters
    ----------
    alpha : float, default=0.5
        Weight for class-conditional distribution vs global prior.
        Must be in [0, 1]. Higher values give more weight to class-specific features.
    
    reg_covar : float, default=1e-6
        Regularization added to diagonal of covariance matrices for numerical stability.
    
    prior_type : str, default='gaussian'
        Type of distribution to model. Options: 'gaussian', 'naive_bayes'
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    
    class_priors_ : ndarray of shape (n_classes,)
        Prior probabilities of each class.
    
    class_means_ : ndarray of shape (n_classes, n_features)
        Mean of each feature for each class.
    
    class_covs_ : ndarray of shape (n_classes, n_features, n_features) or (n_classes, n_features)
        Covariance matrices for each class (full or diagonal).
    
    global_mean_ : ndarray of shape (n_features,)
        Global mean across all samples.
    
    global_cov_ : ndarray of shape (n_features, n_features) or (n_features,)
        Global covariance across all samples.
    """
    
    def __init__(self, alpha=0.5, reg_covar=1e-6, prior_type='gaussian'):
        self.alpha = alpha
        self.reg_covar = reg_covar
        self.prior_type = prior_type
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier by estimating class-conditional and global distributions.
        
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
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        
        # Store classes
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X_train.shape[1]
        
        # Compute class priors
        self.class_priors_ = np.array([
            np.sum(y_train == c) / len(y_train) for c in self.classes_
        ])
        
        # Compute global statistics (shared prior)
        self.global_mean_ = np.mean(X_train, axis=0)
        
        if self.prior_type == 'naive_bayes':
            # Diagonal covariance only
            self.global_cov_ = np.var(X_train, axis=0) + self.reg_covar
            self.class_means_ = np.zeros((self.n_classes_, self.n_features_))
            self.class_covs_ = np.zeros((self.n_classes_, self.n_features_))
            
            for idx, c in enumerate(self.classes_):
                X_c = X_train[y_train == c]
                self.class_means_[idx] = np.mean(X_c, axis=0)
                self.class_covs_[idx] = np.var(X_c, axis=0) + self.reg_covar
        else:
            # Full covariance (Gaussian)
            X_centered = X_train - self.global_mean_
            self.global_cov_ = (X_centered.T @ X_centered) / len(X_train)
            self.global_cov_ += self.reg_covar * np.eye(self.n_features_)
            
            self.class_means_ = np.zeros((self.n_classes_, self.n_features_))
            self.class_covs_ = np.zeros((self.n_classes_, self.n_features_, self.n_features_))
            
            for idx, c in enumerate(self.classes_):
                X_c = X_train[y_train == c]
                self.class_means_[idx] = np.mean(X_c, axis=0)
                X_c_centered = X_c - self.class_means_[idx]
                self.class_covs_[idx] = (X_c_centered.T @ X_c_centered) / len(X_c)
                self.class_covs_[idx] += self.reg_covar * np.eye(self.n_features_)
        
        return self
    
    def _log_likelihood_gaussian(self, X, mean, cov):
        """Compute log likelihood for multivariate Gaussian."""
        n_samples, n_features = X.shape
        X_centered = X - mean
        
        # Compute log determinant and inverse
        sign, logdet = np.linalg.slogdet(cov)
        cov_inv = np.linalg.inv(cov)
        
        # Mahalanobis distance
        mahal = np.sum(X_centered @ cov_inv * X_centered, axis=1)
        
        log_likelihood = -0.5 * (n_features * np.log(2 * np.pi) + logdet + mahal)
        return log_likelihood
    
    def _log_likelihood_naive_bayes(self, X, mean, var):
        """Compute log likelihood for diagonal Gaussian (Naive Bayes)."""
        n_features = X.shape[1]
        X_centered = X - mean
        
        # Log likelihood with diagonal covariance
        log_likelihood = -0.5 * (
            n_features * np.log(2 * np.pi) +
            np.sum(np.log(var)) +
            np.sum((X_centered ** 2) / var, axis=1)
        )
        return log_likelihood
    
    def predict_log_proba(self, X_test):
        """
        Compute log probabilities for each class.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Log probabilities for each class.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        log_proba = np.zeros((n_samples, self.n_classes_))
        
        # Compute global log likelihood (shared prior)
        if self.prior_type == 'naive_bayes':
            log_global = self._log_likelihood_naive_bayes(
                X_test, self.global_mean_, self.global_cov_
            )
        else:
            log_global = self._log_likelihood_gaussian(
                X_test, self.global_mean_, self.global_cov_
            )
        
        # Compute combined log likelihood for each class
        for idx, c in enumerate(self.classes_):
            # Class-conditional log likelihood
            if self.prior_type == 'naive_bayes':
                log_class = self._log_likelihood_naive_bayes(
                    X_test, self.class_means_[idx], self.class_covs_[idx]
                )
            else:
                log_class = self._log_likelihood_gaussian(
                    X_test, self.class_means_[idx], self.class_covs_[idx]
                )
            
            # Linear combination in log space using logsumexp trick
            # log(alpha * p1 + (1-alpha) * p2) = logsumexp([log(alpha) + log(p1), log(1-alpha) + log(p2)])
            log_combined = logsumexp(
                np.column_stack([
                    np.log(self.alpha) + log_class,
                    np.log(1 - self.alpha) + log_global
                ]),
                axis=1
            )
            
            # Add class prior
            log_proba[:, idx] = np.log(self.class_priors_[idx]) + log_combined
        
        # Normalize
        log_proba -= logsumexp(log_proba, axis=1, keepdims=True)
        
        return log_proba
    
    def predict_proba(self, X_test):
        """
        Compute probabilities for each class.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Probabilities for each class.
        """
        return np.exp(self.predict_log_proba(X_test))
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        log_proba = self.predict_log_proba(X_test)
        return self.classes_[np.argmax(log_proba, axis=1)]