import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy import linalg


class FactorAnalysisDiscriminant(BaseEstimator, ClassifierMixin):
    """
    Factor Analysis Discriminant Classifier.
    
    Decomposes each class covariance into shared and class-specific components
    using factor analysis for simplified complexity with retained discriminative power.
    
    The model assumes:
    Σ_k = Λ_k Λ_k^T + Ψ
    where Λ_k is the class-specific factor loading matrix and Ψ is the shared
    diagonal noise covariance matrix.
    
    Parameters
    ----------
    n_factors : int, default=2
        Number of latent factors for each class.
    
    max_iter : int, default=100
        Maximum number of EM iterations for factor analysis.
    
    tol : float, default=1e-4
        Convergence tolerance for EM algorithm.
    
    reg_covar : float, default=1e-6
        Regularization added to diagonal of covariance matrices.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    
    priors_ : ndarray of shape (n_classes,)
        Prior probabilities of each class.
    
    means_ : ndarray of shape (n_classes, n_features)
        Mean vectors for each class.
    
    factor_loadings_ : list of ndarray
        Factor loading matrices for each class.
    
    psi_ : ndarray of shape (n_features,)
        Shared diagonal noise covariance.
    """
    
    def __init__(self, n_factors=2, max_iter=100, tol=1e-4, reg_covar=1e-6):
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
    
    def _fit_factor_analysis(self, X_centered):
        """
        Fit factor analysis model using EM algorithm.
        
        Parameters
        ----------
        X_centered : ndarray of shape (n_samples, n_features)
            Centered data matrix.
        
        Returns
        -------
        Lambda : ndarray of shape (n_features, n_factors)
            Factor loading matrix.
        
        psi : ndarray of shape (n_features,)
            Diagonal noise variance.
        """
        n_samples, n_features = X_centered.shape
        n_factors = min(self.n_factors, n_features)
        
        # Initialize with PCA-like solution
        sample_cov = np.dot(X_centered.T, X_centered) / n_samples
        sample_cov += self.reg_covar * np.eye(n_features)
        
        eigenvalues, eigenvectors = linalg.eigh(sample_cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Initialize Lambda and psi
        Lambda = eigenvectors[:, :n_factors] * np.sqrt(
            np.maximum(eigenvalues[:n_factors] - eigenvalues[n_factors:].mean(), 0.1)
        )
        psi = np.full(n_features, eigenvalues[n_factors:].mean() + self.reg_covar)
        
        # EM algorithm
        for iteration in range(self.max_iter):
            Lambda_old = Lambda.copy()
            
            # E-step: Compute posterior moments
            psi_inv = 1.0 / psi
            Psi_inv = np.diag(psi_inv)
            
            # M = (I + Λ^T Ψ^{-1} Λ)^{-1}
            M = linalg.inv(np.eye(n_factors) + Lambda.T @ Psi_inv @ Lambda)
            
            # E[z|x] = M Λ^T Ψ^{-1} x
            # E[zz^T|x] = M + E[z|x]E[z|x]^T
            Ez = X_centered @ Psi_inv @ Lambda @ M  # (n_samples, n_factors)
            
            # M-step: Update parameters
            # Lambda = (Σ_i x_i E[z_i|x_i]^T) (Σ_i E[z_i z_i^T|x_i])^{-1}
            sum_x_Ez = X_centered.T @ Ez  # (n_features, n_factors)
            sum_Ezz = n_samples * M + Ez.T @ Ez  # (n_factors, n_factors)
            
            Lambda = sum_x_Ez @ linalg.inv(sum_Ezz)
            
            # psi = diag(S - Λ E[Σ_i z_i x_i^T] / n)
            residual = sample_cov - Lambda @ sum_x_Ez.T / n_samples
            psi = np.maximum(np.diag(residual), self.reg_covar)
            
            # Check convergence
            if np.max(np.abs(Lambda - Lambda_old)) < self.tol:
                break
        
        return Lambda, psi
    
    def fit(self, X_train, y_train):
        """
        Fit the Factor Analysis Discriminant classifier.
        
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
        X_train, y_train = check_X_y(X_train, y_train)
        
        self.classes_ = unique_labels(y_train)
        n_classes = len(self.classes_)
        n_features = X_train.shape[1]
        
        # Compute class priors and means
        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        
        # Store centered data for each class
        class_data = []
        
        for idx, cls in enumerate(self.classes_):
            X_cls = X_train[y_train == cls]
            self.priors_[idx] = X_cls.shape[0] / X_train.shape[0]
            self.means_[idx] = np.mean(X_cls, axis=0)
            class_data.append(X_cls - self.means_[idx])
        
        # Fit factor analysis for each class to get class-specific loadings
        self.factor_loadings_ = []
        psi_estimates = []
        
        for X_centered in class_data:
            if X_centered.shape[0] > 1:
                Lambda, psi = self._fit_factor_analysis(X_centered)
                self.factor_loadings_.append(Lambda)
                psi_estimates.append(psi)
            else:
                # Handle edge case with single sample
                Lambda = np.random.randn(n_features, self.n_factors) * 0.1
                psi = np.ones(n_features) * self.reg_covar
                self.factor_loadings_.append(Lambda)
                psi_estimates.append(psi)
        
        # Compute shared diagonal noise covariance (average across classes)
        self.psi_ = np.mean(psi_estimates, axis=0)
        
        return self
    
    def _log_likelihood(self, X, class_idx):
        """
        Compute log-likelihood for a given class.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data samples.
        
        class_idx : int
            Index of the class.
        
        Returns
        -------
        log_likelihood : ndarray of shape (n_samples,)
            Log-likelihood values.
        """
        mean = self.means_[class_idx]
        Lambda = self.factor_loadings_[class_idx]
        
        # Covariance: Σ = Λ Λ^T + Ψ
        # Use Woodbury identity for efficient inverse
        psi_inv = 1.0 / self.psi_
        Psi_inv = np.diag(psi_inv)
        
        # (Σ)^{-1} = Ψ^{-1} - Ψ^{-1} Λ (I + Λ^T Ψ^{-1} Λ)^{-1} Λ^T Ψ^{-1}
        M_inv = np.eye(Lambda.shape[1]) + Lambda.T @ Psi_inv @ Lambda
        M = linalg.inv(M_inv)
        
        # Compute log determinant using matrix determinant lemma
        # log|Σ| = log|Ψ| + log|I + Λ^T Ψ^{-1} Λ|
        log_det = np.sum(np.log(self.psi_)) + np.linalg.slogdet(M_inv)[1]
        
        # Compute Mahalanobis distance
        diff = X - mean
        
        # Efficient computation: x^T Σ^{-1} x
        term1 = np.sum((diff @ Psi_inv) * diff, axis=1)
        temp = diff @ Psi_inv @ Lambda @ M @ Lambda.T @ Psi_inv
        term2 = np.sum(temp * diff, axis=1)
        mahalanobis = term1 - term2
        
        # Log-likelihood
        n_features = X.shape[1]
        log_likelihood = -0.5 * (n_features * np.log(2 * np.pi) + log_det + mahalanobis)
        
        return log_likelihood
    
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
        
        log_proba = np.zeros((n_samples, n_classes))
        
        for idx in range(n_classes):
            log_proba[:, idx] = (
                np.log(self.priors_[idx]) + self._log_likelihood(X_test, idx)
            )
        
        # Normalize to get probabilities (log-sum-exp trick for numerical stability)
        log_proba_max = np.max(log_proba, axis=1, keepdims=True)
        exp_log_proba = np.exp(log_proba - log_proba_max)
        proba = exp_log_proba / np.sum(exp_log_proba, axis=1, keepdims=True)
        
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