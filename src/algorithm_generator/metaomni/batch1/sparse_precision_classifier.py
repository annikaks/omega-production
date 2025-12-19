import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class SparsePrecisionClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that replaces independence assumptions with learned pairwise 
    feature correlations using sparse precision matrices.
    
    This classifier learns a sparse precision (inverse covariance) matrix for 
    each class using Graphical Lasso, capturing feature dependencies while 
    maintaining sparsity for interpretability and computational efficiency.
    
    Parameters
    ----------
    alpha : float or 'auto', default='auto'
        Regularization parameter for GraphicalLasso. Higher values lead to 
        sparser precision matrices. If 'auto', uses cross-validation.
    
    alphas : int or array-like, default=4
        If alpha='auto', this specifies the number of alphas or the grid of 
        alpha values to try in cross-validation.
    
    cv : int, default=5
        Number of cross-validation folds when alpha='auto'.
    
    max_iter : int, default=100
        Maximum number of iterations for GraphicalLasso.
    
    tol : float, default=1e-4
        Convergence tolerance for GraphicalLasso.
    
    assume_centered : bool, default=False
        If True, data are not centered before computation.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    
    class_priors_ : ndarray of shape (n_classes,)
        Prior probabilities of each class.
    
    means_ : dict
        Mean vectors for each class.
    
    precisions_ : dict
        Learned sparse precision matrices for each class.
    
    covariances_ : dict
        Learned covariance matrices for each class.
    
    scaler_ : StandardScaler
        Fitted scaler for feature standardization.
    """
    
    def __init__(self, alpha='auto', alphas=4, cv=5, max_iter=100, 
                 tol=1e-4, assume_centered=False):
        self.alpha = alpha
        self.alphas = alphas
        self.cv = cv
        self.max_iter = max_iter
        self.tol = tol
        self.assume_centered = assume_centered
    
    def fit(self, X, y):
        """
        Fit the Sparse Precision Classifier.
        
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
        
        # Store classes and compute priors
        self.classes_ = unique_labels(y)
        n_samples = X.shape[0]
        
        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Initialize storage
        self.means_ = {}
        self.covariances_ = {}
        self.precisions_ = {}
        self.class_priors_ = np.zeros(len(self.classes_))
        
        # Learn precision matrix for each class
        for idx, class_label in enumerate(self.classes_):
            # Get samples for this class
            X_class = X_scaled[y == class_label]
            n_class_samples = X_class.shape[0]
            
            # Compute prior
            self.class_priors_[idx] = n_class_samples / n_samples
            
            # Compute mean
            self.means_[class_label] = np.mean(X_class, axis=0)
            
            # Learn sparse precision matrix using Graphical Lasso
            if n_class_samples > 1:
                if self.alpha == 'auto':
                    # Use cross-validation to select alpha
                    model = GraphicalLassoCV(
                        alphas=self.alphas,
                        cv=self.cv,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        assume_centered=self.assume_centered,
                        n_jobs=-1
                    )
                else:
                    from sklearn.covariance import GraphicalLasso
                    model = GraphicalLasso(
                        alpha=self.alpha,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        assume_centered=self.assume_centered
                    )
                
                try:
                    model.fit(X_class)
                    self.covariances_[class_label] = model.covariance_
                    self.precisions_[class_label] = model.precision_
                except Exception as e:
                    # Fallback to diagonal covariance if GraphicalLasso fails
                    print(f"Warning: GraphicalLasso failed for class {class_label}. "
                          f"Using diagonal covariance. Error: {e}")
                    cov = np.diag(np.var(X_class, axis=0) + 1e-6)
                    self.covariances_[class_label] = cov
                    self.precisions_[class_label] = np.linalg.inv(cov)
            else:
                # Single sample: use identity
                n_features = X_class.shape[1]
                self.covariances_[class_label] = np.eye(n_features)
                self.precisions_[class_label] = np.eye(n_features)
        
        return self
    
    def _log_likelihood(self, X, class_label):
        """
        Compute log-likelihood of X under the Gaussian distribution for a class.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        
        class_label : int or str
            Class label.
        
        Returns
        -------
        log_likelihood : ndarray of shape (n_samples,)
            Log-likelihood for each sample.
        """
        mean = self.means_[class_label]
        precision = self.precisions_[class_label]
        covariance = self.covariances_[class_label]
        
        # Center the data
        X_centered = X - mean
        
        # Compute log determinant of covariance
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            logdet = np.linalg.slogdet(covariance + 1e-6 * np.eye(covariance.shape[0]))[1]
        
        # Compute Mahalanobis distance using precision matrix
        # (x - mu)^T @ Precision @ (x - mu)
        mahalanobis = np.sum(X_centered @ precision * X_centered, axis=1)
        
        # Log-likelihood: -0.5 * (k*log(2*pi) + log|Sigma| + mahalanobis)
        k = X.shape[1]
        log_likelihood = -0.5 * (k * np.log(2 * np.pi) + logdet + mahalanobis)
        
        return log_likelihood
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # Standardize features
        X_scaled = self.scaler_.transform(X)
        
        # Compute log posterior for each class
        n_samples = X_scaled.shape[0]
        n_classes = len(self.classes_)
        log_posteriors = np.zeros((n_samples, n_classes))
        
        for idx, class_label in enumerate(self.classes_):
            # Log posterior = log prior + log likelihood
            log_prior = np.log(self.class_priors_[idx] + 1e-10)
            log_likelihood = self._log_likelihood(X_scaled, class_label)
            log_posteriors[:, idx] = log_prior + log_likelihood
        
        # Convert log posteriors to probabilities using log-sum-exp trick
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
            Input samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def get_feature_correlations(self, class_label):
        """
        Get the learned feature correlation structure for a specific class.
        
        Parameters
        ----------
        class_label : int or str
            Class label.
        
        Returns
        -------
        precision : ndarray of shape (n_features, n_features)
            Sparse precision matrix encoding feature correlations.
        """
        check_is_fitted(self)
        if class_label not in self.precisions_:
            raise ValueError(f"Class {class_label} not found in fitted classes.")
        return self.precisions_[class_label]