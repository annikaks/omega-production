import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import GraphicalLassoCV
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.linalg import det, solve
import warnings


class SparseGaussianClassifier(BaseEstimator, ClassifierMixin):
    """
    Gaussian classifier using sparse precision matrices via graphical lasso.
    
    This classifier exploits conditional independence structure by estimating
    sparse precision (inverse covariance) matrices using graphical lasso,
    which is more efficient and robust than full covariance matrices,
    especially in high-dimensional settings.
    
    Parameters
    ----------
    alpha : float or array-like of shape (n_alphas,), default=None
        Regularization parameter for graphical lasso. If None, uses cross-validation.
        Higher values lead to sparser precision matrices.
    
    cv : int, default=5
        Number of cross-validation folds for selecting alpha (if alpha is None).
    
    assume_centered : bool, default=False
        If True, data are not centered before computation.
    
    max_iter : int, default=100
        Maximum number of iterations for graphical lasso.
    
    tol : float, default=1e-4
        Convergence tolerance for graphical lasso.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    
    class_priors_ : ndarray of shape (n_classes,)
        Prior probabilities of each class.
    
    means_ : dict
        Mean vectors for each class.
    
    precisions_ : dict
        Sparse precision matrices for each class.
    
    log_det_precisions_ : dict
        Log determinants of precision matrices for each class.
    """
    
    def __init__(self, alpha=None, cv=5, assume_centered=False, 
                 max_iter=100, tol=1e-4):
        self.alpha = alpha
        self.cv = cv
        self.assume_centered = assume_centered
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X, y):
        """
        Fit the sparse Gaussian classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input
        X, y = check_X_y(X, y)
        
        # Store classes
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        # Initialize storage
        self.means_ = {}
        self.precisions_ = {}
        self.log_det_precisions_ = {}
        self.class_priors_ = np.zeros(n_classes)
        
        # Fit a separate model for each class
        for idx, class_label in enumerate(self.classes_):
            # Get samples for this class
            X_class = X[y == class_label]
            n_class_samples = X_class.shape[0]
            
            # Store prior probability
            self.class_priors_[idx] = n_class_samples / n_samples
            
            # Compute mean
            self.means_[class_label] = np.mean(X_class, axis=0)
            
            # Center the data
            X_centered = X_class - self.means_[class_label]
            
            # Handle edge cases
            if n_class_samples < 2:
                warnings.warn(
                    f"Class {class_label} has only {n_class_samples} sample(s). "
                    "Using identity precision matrix."
                )
                self.precisions_[class_label] = np.eye(n_features)
                self.log_det_precisions_[class_label] = 0.0
                continue
            
            # Estimate sparse precision matrix using graphical lasso
            try:
                if self.alpha is None:
                    # Use cross-validation to select alpha
                    glasso = GraphicalLassoCV(
                        cv=self.cv,
                        assume_centered=self.assume_centered,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        n_jobs=1
                    )
                else:
                    from sklearn.covariance import GraphicalLasso
                    glasso = GraphicalLasso(
                        alpha=self.alpha,
                        assume_centered=self.assume_centered,
                        max_iter=self.max_iter,
                        tol=self.tol
                    )
                
                glasso.fit(X_centered)
                precision = glasso.precision_
                
                # Store precision matrix
                self.precisions_[class_label] = precision
                
                # Compute log determinant for likelihood calculation
                # Use sign and logdet for numerical stability
                sign, logdet = np.linalg.slogdet(precision)
                if sign <= 0:
                    warnings.warn(
                        f"Precision matrix for class {class_label} is not positive definite. "
                        "Using regularized version."
                    )
                    # Add small regularization to diagonal
                    precision_reg = precision + 1e-6 * np.eye(n_features)
                    self.precisions_[class_label] = precision_reg
                    sign, logdet = np.linalg.slogdet(precision_reg)
                
                self.log_det_precisions_[class_label] = logdet
                
            except Exception as e:
                warnings.warn(
                    f"Graphical lasso failed for class {class_label}: {str(e)}. "
                    "Using empirical covariance with regularization."
                )
                # Fallback to regularized empirical covariance
                cov = np.cov(X_centered.T) + 1e-6 * np.eye(n_features)
                precision = np.linalg.inv(cov)
                self.precisions_[class_label] = precision
                sign, logdet = np.linalg.slogdet(precision)
                self.log_det_precisions_[class_label] = logdet
        
        return self
    
    def _compute_log_likelihood(self, X, class_label):
        """
        Compute log likelihood for a given class.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        class_label : int or str
            Class label.
        
        Returns
        -------
        log_likelihood : ndarray of shape (n_samples,)
            Log likelihood for each sample.
        """
        mean = self.means_[class_label]
        precision = self.precisions_[class_label]
        log_det_precision = self.log_det_precisions_[class_label]
        
        # Center the data
        X_centered = X - mean
        
        # Compute quadratic form: (x - mu)^T * Precision * (x - mu)
        # Using einsum for efficiency
        quadratic_form = np.einsum('ij,jk,ik->i', X_centered, precision, X_centered)
        
        # Log likelihood (without constant term)
        # log p(x|class) = -0.5 * [(x-mu)^T * Precision * (x-mu) - log|Precision|]
        # We include log|Precision| as it's the log determinant of precision matrix
        n_features = X.shape[1]
        log_likelihood = (
            0.5 * log_det_precision 
            - 0.5 * quadratic_form 
            - 0.5 * n_features * np.log(2 * np.pi)
        )
        
        return log_likelihood
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Compute log posterior for each class
        log_posteriors = np.zeros((n_samples, n_classes))
        
        for idx, class_label in enumerate(self.classes_):
            log_likelihood = self._compute_log_likelihood(X, class_label)
            log_prior = np.log(self.class_priors_[idx])
            log_posteriors[:, idx] = log_likelihood + log_prior
        
        # Convert log posteriors to probabilities using log-sum-exp trick
        # for numerical stability
        log_posteriors_max = np.max(log_posteriors, axis=1, keepdims=True)
        posteriors = np.exp(log_posteriors - log_posteriors_max)
        posteriors /= np.sum(posteriors, axis=1, keepdims=True)
        
        return posteriors
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def predict_log_proba(self, X):
        """
        Predict log probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Log probabilities.
        """
        proba = self.predict_proba(X)
        return np.log(proba + 1e-10)  # Add small constant to avoid log(0)