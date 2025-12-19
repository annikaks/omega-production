import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import multivariate_normal


class RegularizedCovarianceClassifier(BaseEstimator, ClassifierMixin):
    """
    Regularized Covariance Classifier with class-specific shrinkage intensities.
    
    This classifier uses Ledoit-Wolf shrinkage to regularize covariance matrices
    for each class, with shrinkage intensities learned via cross-validation to
    balance the bias-variance tradeoff per class.
    
    Parameters
    ----------
    shrinkage_grid : array-like, default=None
        Grid of shrinkage intensities to search over. If None, uses
        np.linspace(0, 1, 21).
    cv : int, default=5
        Number of cross-validation folds for selecting shrinkage intensity.
    priors : array-like, default=None
        Prior probabilities of classes. If None, uses empirical class frequencies.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    class_means_ : dict
        Mean vectors for each class.
    class_covs_ : dict
        Regularized covariance matrices for each class.
    shrinkage_intensities_ : dict
        Optimal shrinkage intensity for each class.
    priors_ : dict
        Prior probabilities for each class.
    """
    
    def __init__(self, shrinkage_grid=None, cv=5, priors=None):
        self.shrinkage_grid = shrinkage_grid
        self.cv = cv
        self.priors = priors
    
    def _compute_shrinkage_target(self, cov):
        """Compute shrinkage target (diagonal matrix with average variance)."""
        return np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    
    def _shrink_covariance(self, cov, shrinkage):
        """Apply shrinkage to covariance matrix."""
        target = self._compute_shrinkage_target(cov)
        return (1 - shrinkage) * cov + shrinkage * target
    
    def _compute_covariance(self, X):
        """Compute sample covariance matrix."""
        X_centered = X - np.mean(X, axis=0)
        n_samples = X.shape[0]
        if n_samples > 1:
            cov = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        else:
            cov = np.eye(X.shape[1]) * 1e-6
        # Add small regularization for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6
        return cov
    
    def _log_likelihood(self, X, mean, cov):
        """Compute log-likelihood of data under Gaussian distribution."""
        try:
            # Use multivariate normal for numerical stability
            rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
            return np.sum(rv.logpdf(X))
        except:
            # Fallback to manual computation
            n_features = X.shape[1]
            X_centered = X - mean
            try:
                cov_inv = np.linalg.inv(cov)
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    logdet = np.log(np.linalg.det(cov + np.eye(n_features) * 1e-3))
            except:
                cov_inv = np.linalg.pinv(cov)
                logdet = np.log(np.linalg.det(cov + np.eye(n_features) * 1e-3))
            
            mahalanobis = np.sum(X_centered @ cov_inv * X_centered, axis=1)
            return -0.5 * np.sum(mahalanobis + logdet + n_features * np.log(2 * np.pi))
    
    def _select_shrinkage_cv(self, X, y, class_label):
        """Select optimal shrinkage intensity via cross-validation for a class."""
        shrinkage_grid = self.shrinkage_grid
        if shrinkage_grid is None:
            shrinkage_grid = np.linspace(0, 1, 21)
        
        # Get data for this class
        X_class = X[y == class_label]
        
        if len(X_class) < self.cv:
            # Not enough samples for CV, use default shrinkage
            return 0.5
        
        cv_scores = np.zeros(len(shrinkage_grid))
        
        skf = StratifiedKFold(n_splits=min(self.cv, len(X_class)), shuffle=True, random_state=42)
        # Create dummy labels for stratification (all same class)
        dummy_labels = np.zeros(len(X_class))
        
        for shrinkage_idx, shrinkage in enumerate(shrinkage_grid):
            fold_scores = []
            
            for train_idx, val_idx in skf.split(X_class, dummy_labels):
                if len(train_idx) < 2:
                    continue
                    
                X_train_fold = X_class[train_idx]
                X_val_fold = X_class[val_idx]
                
                # Compute mean and covariance on training fold
                mean_fold = np.mean(X_train_fold, axis=0)
                cov_fold = self._compute_covariance(X_train_fold)
                cov_shrunk = self._shrink_covariance(cov_fold, shrinkage)
                
                # Evaluate on validation fold
                ll = self._log_likelihood(X_val_fold, mean_fold, cov_shrunk)
                fold_scores.append(ll)
            
            if fold_scores:
                cv_scores[shrinkage_idx] = np.mean(fold_scores)
            else:
                cv_scores[shrinkage_idx] = -np.inf
        
        # Select shrinkage with best CV score
        best_idx = np.argmax(cv_scores)
        return shrinkage_grid[best_idx]
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier.
        
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
        X_train, y_train = check_X_y(X_train, y_train)
        self.classes_ = unique_labels(y_train)
        
        self.class_means_ = {}
        self.class_covs_ = {}
        self.shrinkage_intensities_ = {}
        self.priors_ = {}
        
        n_samples = len(y_train)
        
        for class_label in self.classes_:
            X_class = X_train[y_train == class_label]
            
            # Compute class prior
            if self.priors is None:
                self.priors_[class_label] = len(X_class) / n_samples
            else:
                self.priors_[class_label] = self.priors[class_label]
            
            # Compute mean
            self.class_means_[class_label] = np.mean(X_class, axis=0)
            
            # Select optimal shrinkage via CV
            optimal_shrinkage = self._select_shrinkage_cv(X_train, y_train, class_label)
            self.shrinkage_intensities_[class_label] = optimal_shrinkage
            
            # Compute regularized covariance
            cov = self._compute_covariance(X_class)
            self.class_covs_[class_label] = self._shrink_covariance(cov, optimal_shrinkage)
        
        return self
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities.
        
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
        log_proba = np.zeros((n_samples, n_classes))
        
        for idx, class_label in enumerate(self.classes_):
            mean = self.class_means_[class_label]
            cov = self.class_covs_[class_label]
            prior = self.priors_[class_label]
            
            try:
                rv = multivariate_normal(mean=mean, cov=cov, allow_singular=True)
                log_proba[:, idx] = rv.logpdf(X_test) + np.log(prior)
            except:
                # Fallback computation
                X_centered = X_test - mean
                cov_inv = np.linalg.pinv(cov)
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    logdet = np.log(np.linalg.det(cov + np.eye(cov.shape[0]) * 1e-3))
                
                mahalanobis = np.sum(X_centered @ cov_inv * X_centered, axis=1)
                log_proba[:, idx] = (-0.5 * (mahalanobis + logdet + 
                                     X_test.shape[1] * np.log(2 * np.pi)) + 
                                     np.log(prior))
        
        # Convert log probabilities to probabilities
        log_proba_max = np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba - log_proba_max)
        proba /= np.sum(proba, axis=1, keepdims=True)
        
        return proba
    
    def predict(self, X_test):
        """
        Predict class labels.
        
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