import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.special import digamma, betaln
from scipy.stats import beta


class BetaBernoulliClassifier(BaseEstimator, ClassifierMixin):
    """
    Bayesian classifier using Beta distributions for Bernoulli parameter uncertainty.
    
    This classifier models feature probabilities using Beta distributions instead of
    point estimates, enabling uncertainty quantification in predictions.
    
    Parameters
    ----------
    alpha_prior : float, default=1.0
        Prior pseudo-count for positive outcomes (Beta distribution alpha parameter).
        
    beta_prior : float, default=1.0
        Prior pseudo-count for negative outcomes (Beta distribution beta parameter).
        
    sampling_method : str, default='mean'
        Method for prediction: 'mean' uses expected values, 'sample' draws from posteriors.
        
    n_samples : int, default=100
        Number of samples to draw when sampling_method='sample'.
        
    binarize_threshold : float, default=0.5
        Threshold for binarizing continuous features.
        
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
        
    class_prior_ : ndarray of shape (n_classes,)
        Prior probabilities of each class.
        
    alpha_params_ : ndarray of shape (n_classes, n_features)
        Alpha parameters of Beta distributions for each feature and class.
        
    beta_params_ : ndarray of shape (n_classes, n_features)
        Beta parameters of Beta distributions for each feature and class.
        
    n_features_in_ : int
        Number of features seen during fit.
    """
    
    def __init__(self, alpha_prior=1.0, beta_prior=1.0, sampling_method='mean',
                 n_samples=100, binarize_threshold=0.5):
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.sampling_method = sampling_method
        self.n_samples = n_samples
        self.binarize_threshold = binarize_threshold
    
    def _binarize(self, X):
        """Binarize features if they are not already binary."""
        return (X > self.binarize_threshold).astype(float)
    
    def fit(self, X, y):
        """
        Fit the Beta-Bernoulli classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
            
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Binarize features
        X_binary = self._binarize(X)
        
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Initialize parameters
        self.alpha_params_ = np.zeros((n_classes, n_features))
        self.beta_params_ = np.zeros((n_classes, n_features))
        self.class_counts_ = np.zeros(n_classes)
        
        # Compute posterior parameters for each class
        for idx, c in enumerate(self.classes_):
            X_c = X_binary[y == c]
            n_samples_c = X_c.shape[0]
            self.class_counts_[idx] = n_samples_c
            
            # Count positive occurrences for each feature
            positive_counts = np.sum(X_c, axis=0)
            negative_counts = n_samples_c - positive_counts
            
            # Update Beta distribution parameters (posterior)
            self.alpha_params_[idx] = self.alpha_prior + positive_counts
            self.beta_params_[idx] = self.beta_prior + negative_counts
        
        # Compute class priors
        self.class_prior_ = self.class_counts_ / np.sum(self.class_counts_)
        
        return self
    
    def _compute_log_likelihood_mean(self, X):
        """
        Compute log likelihood using expected values of Beta distributions.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
            
        Returns
        -------
        log_likelihood : ndarray of shape (n_samples, n_classes)
            Log likelihood for each sample and class.
        """
        X_binary = self._binarize(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        log_likelihood = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            # Expected value of Beta distribution: alpha / (alpha + beta)
            theta_mean = self.alpha_params_[idx] / (
                self.alpha_params_[idx] + self.beta_params_[idx]
            )
            
            # Clip to avoid log(0)
            theta_mean = np.clip(theta_mean, 1e-10, 1 - 1e-10)
            
            # Compute log likelihood for Bernoulli distribution
            log_likelihood[:, idx] = np.sum(
                X_binary * np.log(theta_mean) + 
                (1 - X_binary) * np.log(1 - theta_mean),
                axis=1
            )
        
        return log_likelihood
    
    def _compute_log_likelihood_sample(self, X):
        """
        Compute log likelihood by sampling from Beta distributions.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
            
        Returns
        -------
        log_likelihood : ndarray of shape (n_samples, n_classes)
            Average log likelihood for each sample and class.
        """
        X_binary = self._binarize(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        log_likelihood_samples = np.zeros((self.n_samples, n_samples, n_classes))
        
        for sample_idx in range(self.n_samples):
            for class_idx in range(n_classes):
                # Sample theta from Beta distribution
                theta_sample = np.random.beta(
                    self.alpha_params_[class_idx],
                    self.beta_params_[class_idx]
                )
                
                # Clip to avoid log(0)
                theta_sample = np.clip(theta_sample, 1e-10, 1 - 1e-10)
                
                # Compute log likelihood
                log_likelihood_samples[sample_idx, :, class_idx] = np.sum(
                    X_binary * np.log(theta_sample) + 
                    (1 - X_binary) * np.log(1 - theta_sample),
                    axis=1
                )
        
        # Average over samples
        log_likelihood = np.mean(log_likelihood_samples, axis=0)
        
        return log_likelihood
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities for each sample.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        
        # Compute log likelihood
        if self.sampling_method == 'mean':
            log_likelihood = self._compute_log_likelihood_mean(X)
        elif self.sampling_method == 'sample':
            log_likelihood = self._compute_log_likelihood_sample(X)
        else:
            raise ValueError(f"Unknown sampling_method: {self.sampling_method}")
        
        # Add log prior
        log_posterior = log_likelihood + np.log(self.class_prior_)
        
        # Convert to probabilities using log-sum-exp trick for numerical stability
        log_posterior_max = np.max(log_posterior, axis=1, keepdims=True)
        posterior = np.exp(log_posterior - log_posterior_max)
        posterior /= np.sum(posterior, axis=1, keepdims=True)
        
        return posterior
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def get_uncertainty(self, X):
        """
        Get prediction uncertainty for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
            
        Returns
        -------
        uncertainty : ndarray of shape (n_samples,)
            Entropy-based uncertainty measure for each sample.
        """
        proba = self.predict_proba(X)
        # Compute entropy as uncertainty measure
        entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        return entropy
    
    def get_feature_uncertainty(self):
        """
        Get uncertainty in feature parameters.
        
        Returns
        -------
        uncertainty : ndarray of shape (n_classes, n_features)
            Variance of Beta distributions for each feature and class.
        """
        check_is_fitted(self)
        
        # Variance of Beta distribution: (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        alpha = self.alpha_params_
        beta_param = self.beta_params_
        
        variance = (alpha * beta_param) / (
            (alpha + beta_param) ** 2 * (alpha + beta_param + 1)
        )
        
        return variance