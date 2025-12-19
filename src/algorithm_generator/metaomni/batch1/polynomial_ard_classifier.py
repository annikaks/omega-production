import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.special import expit


class PolynomialARDClassifier(BaseEstimator, ClassifierMixin):
    """
    Polynomial interaction classifier with Automatic Relevance Determination (ARD).
    
    This classifier adds polynomial interaction terms between features and uses
    ARD to automatically determine feature relevance through Bayesian regularization.
    Supports both binary and multi-class classification (via One-vs-Rest).
    
    Parameters
    ----------
    degree : int, default=2
        Degree of polynomial features to generate.
    
    max_iter : int, default=300
        Maximum number of iterations for optimization.
    
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    
    alpha_init : float, default=1.0
        Initial value for precision of weights (regularization).
    
    beta_init : float, default=1.0
        Initial value for noise precision.
    
    threshold_alpha : float, default=1e6
        Threshold for pruning features with high precision (low relevance).
    
    include_bias : bool, default=True
        Whether to include bias term.
    
    interaction_only : bool, default=False
        If True, only interaction features are produced.
    """
    
    def __init__(
        self,
        degree=2,
        max_iter=300,
        tol=1e-3,
        alpha_init=1.0,
        beta_init=1.0,
        threshold_alpha=1e6,
        include_bias=True,
        interaction_only=False
    ):
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.threshold_alpha = threshold_alpha
        self.include_bias = include_bias
        self.interaction_only = interaction_only
    
    def _create_polynomial_features(self, X):
        """Generate polynomial features."""
        poly = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )
        return poly.fit_transform(X), poly
    
    def _sigmoid(self, X):
        """Sigmoid activation function."""
        return expit(X)
    
    def _fit_ard_binary(self, X, y):
        """Fit using ARD with iterative reweighting for binary classification."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        alpha = np.full(n_features, self.alpha_init)
        beta = self.beta_init
        weights = np.zeros(n_features)
        
        # Keep track of relevant features
        relevant_features = np.ones(n_features, dtype=bool)
        
        for iteration in range(self.max_iter):
            weights_old = weights.copy()
            
            # E-step: Compute posterior mean and covariance
            # For logistic regression, use iterative reweighted least squares (IRLS)
            predictions = self._sigmoid(X[:, relevant_features] @ weights[relevant_features])
            
            # Compute weights for IRLS
            W = predictions * (1 - predictions) + 1e-10
            
            # Compute Hessian (precision matrix)
            A = np.diag(alpha[relevant_features])
            H = X[:, relevant_features].T @ (W[:, np.newaxis] * X[:, relevant_features]) + A
            
            # Add small regularization for numerical stability
            H += np.eye(H.shape[0]) * 1e-10
            
            try:
                # Compute covariance
                Sigma = np.linalg.inv(H)
                
                # Compute gradient
                grad = X[:, relevant_features].T @ (y - predictions) - A @ weights[relevant_features]
                
                # Update weights
                weights[relevant_features] += Sigma @ grad
                
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse
                Sigma = np.linalg.pinv(H)
                grad = X[:, relevant_features].T @ (y - predictions) - A @ weights[relevant_features]
                weights[relevant_features] += Sigma @ grad
            
            # M-step: Update hyperparameters (alpha)
            gamma = 1 - alpha[relevant_features] * np.diag(Sigma)
            alpha[relevant_features] = gamma / (weights[relevant_features]**2 + 1e-10)
            
            # Prune features with very high alpha (low relevance)
            relevant_features_new = alpha < self.threshold_alpha
            
            if not np.array_equal(relevant_features, relevant_features_new):
                # Reset weights for pruned features
                weights[~relevant_features_new] = 0
                relevant_features = relevant_features_new
                
                if np.sum(relevant_features) == 0:
                    # Keep at least one feature
                    min_alpha_idx = np.argmin(alpha)
                    relevant_features[min_alpha_idx] = True
                    weights[min_alpha_idx] = weights_old[min_alpha_idx]
            
            # Check convergence
            weight_change = np.max(np.abs(weights - weights_old))
            if weight_change < self.tol:
                break
        
        return weights, alpha, relevant_features, iteration + 1
    
    def fit(self, X, y):
        """
        Fit the polynomial ARD classifier.
        
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
        # Check and validate input
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Generate polynomial features
        X_poly, self.poly_transformer_ = self._create_polynomial_features(X)
        
        n_classes = len(self.classes_)
        
        if n_classes == 2:
            # Binary classification
            y_binary = (y == self.classes_[1]).astype(float)
            weights, alpha, relevant_features, n_iter = self._fit_ard_binary(X_poly, y_binary)
            
            self.weights_ = weights
            self.alpha_ = alpha
            self.relevant_features_ = relevant_features
            self.n_iter_ = n_iter
            self.is_multiclass_ = False
            
        else:
            # Multi-class classification using One-vs-Rest
            self.weights_ = []
            self.alpha_ = []
            self.relevant_features_ = []
            self.n_iter_ = []
            
            for class_idx, class_label in enumerate(self.classes_):
                # Create binary target for this class
                y_binary = (y == class_label).astype(float)
                
                # Fit binary classifier
                weights, alpha, relevant_features, n_iter = self._fit_ard_binary(X_poly, y_binary)
                
                self.weights_.append(weights)
                self.alpha_.append(alpha)
                self.relevant_features_.append(relevant_features)
                self.n_iter_.append(n_iter)
            
            self.is_multiclass_ = True
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # Transform to polynomial features
        X_poly = self.poly_transformer_.transform(X)
        
        if not self.is_multiclass_:
            # Binary classification
            logits = X_poly @ self.weights_
            proba_class1 = self._sigmoid(logits)
            proba_class0 = 1 - proba_class1
            return np.column_stack([proba_class0, proba_class1])
        else:
            # Multi-class classification
            n_samples = X_poly.shape[0]
            n_classes = len(self.classes_)
            
            # Compute decision function for each class
            decision_values = np.zeros((n_samples, n_classes))
            for class_idx in range(n_classes):
                logits = X_poly @ self.weights_[class_idx]
                decision_values[:, class_idx] = self._sigmoid(logits)
            
            # Normalize to get probabilities
            proba = decision_values / (decision_values.sum(axis=1, keepdims=True) + 1e-10)
            
            return proba
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def get_feature_relevance(self):
        """
        Get relevance scores for polynomial features.
        
        Returns
        -------
        relevance : dict or list of dict
            Dictionary (binary) or list of dictionaries (multi-class) with 
            feature indices, weights, and alpha values.
        """
        check_is_fitted(self)
        
        if not self.is_multiclass_:
            return {
                'relevant_indices': np.where(self.relevant_features_)[0],
                'weights': self.weights_[self.relevant_features_],
                'alpha': self.alpha_[self.relevant_features_],
                'n_relevant': np.sum(self.relevant_features_),
                'n_total': len(self.relevant_features_)
            }
        else:
            relevance_list = []
            for class_idx in range(len(self.classes_)):
                relevance_list.append({
                    'class': self.classes_[class_idx],
                    'relevant_indices': np.where(self.relevant_features_[class_idx])[0],
                    'weights': self.weights_[class_idx][self.relevant_features_[class_idx]],
                    'alpha': self.alpha_[class_idx][self.relevant_features_[class_idx]],
                    'n_relevant': np.sum(self.relevant_features_[class_idx]),
                    'n_total': len(self.relevant_features_[class_idx])
                })
            return relevance_list