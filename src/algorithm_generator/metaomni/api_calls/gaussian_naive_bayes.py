import numpy as np
from sklearn.base import BaseEstimator


class GaussianNaiveBayes(BaseEstimator):
    def __init__(self, var_smoothing=1e-9):
        """
        Gaussian Naive Bayes classifier.
        
        Parameters
        ----------
        var_smoothing : float, default=1e-9
            Portion of the largest variance of all features that is added to
            variances for calculation stability.
        """
        self.var_smoothing = var_smoothing
    
    def fit(self, X_train, y_train):
        """
        Fit Gaussian Naive Bayes classifier.
        
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
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        self.classes_ = np.unique(y_train)
        n_classes = len(self.classes_)
        n_features = X_train.shape[1]
        
        # Initialize arrays to store statistics
        self.class_prior_ = np.zeros(n_classes)
        self.theta_ = np.zeros((n_classes, n_features))  # means
        self.var_ = np.zeros((n_classes, n_features))    # variances
        
        # Calculate statistics for each class
        for idx, c in enumerate(self.classes_):
            X_c = X_train[y_train == c]
            self.class_prior_[idx] = X_c.shape[0] / X_train.shape[0]
            self.theta_[idx, :] = X_c.mean(axis=0)
            self.var_[idx, :] = X_c.var(axis=0)
        
        # Add smoothing to variances
        self.var_ += self.var_smoothing * np.var(X_train, axis=0).max()
        
        return self
    
    def _calculate_log_likelihood(self, X):
        """
        Calculate log likelihood using Gaussian probability density function.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        log_likelihood : array of shape (n_samples, n_classes)
            Log likelihood for each sample and class.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        log_likelihood = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            # Log of Gaussian PDF: log(1/sqrt(2*pi*var)) - (x-mu)^2/(2*var)
            log_prior = np.log(self.class_prior_[idx])
            log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[idx, :]))
            log_prob -= 0.5 * np.sum(
                ((X - self.theta_[idx, :]) ** 2) / self.var_[idx, :], axis=1
            )
            log_likelihood[:, idx] = log_prior + log_prob
        
        return log_likelihood
    
    def predict(self, X_test):
        """
        Perform classification on test data.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        X_test = np.asarray(X_test)
        log_likelihood = self._calculate_log_likelihood(X_test)
        return self.classes_[np.argmax(log_likelihood, axis=1)]
    
    def predict_proba(self, X_test):
        """
        Return probability estimates for the test data.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Probability estimates.
        """
        X_test = np.asarray(X_test)
        log_likelihood = self._calculate_log_likelihood(X_test)
        
        # Convert log likelihoods to probabilities using log-sum-exp trick
        log_likelihood_max = np.max(log_likelihood, axis=1, keepdims=True)
        exp_log_likelihood = np.exp(log_likelihood - log_likelihood_max)
        proba = exp_log_likelihood / np.sum(exp_log_likelihood, axis=1, keepdims=True)
        
        return proba
    
    def predict_log_proba(self, X_test):
        """
        Return log-probability estimates for the test data.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        log_proba : array of shape (n_samples, n_classes)
            Log-probability estimates.
        """
        X_test = np.asarray(X_test)
        log_likelihood = self._calculate_log_likelihood(X_test)
        
        # Normalize log likelihoods using log-sum-exp trick
        log_likelihood_max = np.max(log_likelihood, axis=1, keepdims=True)
        log_sum_exp = log_likelihood_max + np.log(
            np.sum(np.exp(log_likelihood - log_likelihood_max), axis=1, keepdims=True)
        )
        log_proba = log_likelihood - log_sum_exp
        
        return log_proba