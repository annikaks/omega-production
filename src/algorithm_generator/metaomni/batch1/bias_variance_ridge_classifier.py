import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class BiasVarianceRidgeClassifier(BaseEstimator, ClassifierMixin):
    """
    Ridge classifier with separate bias and variance regularization.
    
    The ridge penalty is decomposed into:
    - Lambda_bias: regularization for the bias term (intercept)
    - Lambda_variance: regularization for the feature weights
    
    Parameters
    ----------
    lambda_bias : float, default=1.0
        Regularization strength for bias term
    lambda_variance : float, default=1.0
        Regularization strength for variance (weights)
    cv : int, default=5
        Number of cross-validation folds for parameter optimization
    param_grid : dict, optional
        Grid of parameters to search. If None, uses default grid.
    fit_intercept : bool, default=True
        Whether to fit an intercept term
    max_iter : int, default=1000
        Maximum number of iterations for optimization
    tol : float, default=1e-4
        Tolerance for optimization convergence
    """
    
    def __init__(self, lambda_bias=1.0, lambda_variance=1.0, cv=5, 
                 param_grid=None, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.lambda_bias = lambda_bias
        self.lambda_variance = lambda_variance
        self.cv = cv
        self.param_grid = param_grid
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
    
    def _ridge_solve(self, X, y, lambda_bias, lambda_variance):
        """
        Solve ridge regression with separate bias and variance penalties.
        
        Minimizes: ||y - X*w - b||^2 + lambda_variance*||w||^2 + lambda_bias*b^2
        """
        n_samples, n_features = X.shape
        
        if self.fit_intercept:
            # Augment X with a column of ones for the intercept
            X_aug = np.column_stack([np.ones(n_samples), X])
            
            # Create regularization matrix with different penalties
            reg_matrix = np.diag(np.concatenate([[lambda_bias], 
                                                  np.full(n_features, lambda_variance)]))
            
            # Solve: (X^T X + R) w = X^T y
            XtX = X_aug.T @ X_aug
            Xty = X_aug.T @ y
            
            try:
                coef_aug = np.linalg.solve(XtX + reg_matrix, Xty)
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                coef_aug = np.linalg.lstsq(XtX + reg_matrix, Xty, rcond=None)[0]
            
            intercept = coef_aug[0]
            coef = coef_aug[1:]
        else:
            # No intercept case
            reg_matrix = lambda_variance * np.eye(n_features)
            XtX = X.T @ X
            Xty = X.T @ y
            
            try:
                coef = np.linalg.solve(XtX + reg_matrix, Xty)
            except np.linalg.LinAlgError:
                coef = np.linalg.lstsq(XtX + reg_matrix, Xty, rcond=None)[0]
            
            intercept = 0.0
        
        return coef, intercept
    
    def _fit_binary(self, X, y, lambda_bias, lambda_variance):
        """Fit binary classification problem."""
        coef, intercept = self._ridge_solve(X, y, lambda_bias, lambda_variance)
        return coef, intercept
    
    def fit(self, X, y):
        """
        Fit the bias-variance ridge classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Validate input
        X, y = check_X_y(X, y)
        
        # Store classes
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        # Encode labels
        self.label_binarizer_ = LabelBinarizer(pos_label=1, neg_label=-1)
        y_encoded = self.label_binarizer_.fit_transform(y)
        
        if self.n_classes_ == 2:
            y_encoded = y_encoded.ravel()
        
        # Optimize hyperparameters via cross-validation if param_grid provided
        if self.param_grid is not None:
            self._optimize_parameters(X, y_encoded)
        
        # Fit the model with optimized or provided parameters
        if self.n_classes_ == 2:
            # Binary classification
            self.coef_, self.intercept_ = self._fit_binary(
                X, y_encoded, self.lambda_bias, self.lambda_variance
            )
            self.coef_ = self.coef_.reshape(1, -1)
            self.intercept_ = np.array([self.intercept_])
        else:
            # Multi-class classification (one-vs-rest)
            coefs = []
            intercepts = []
            for i in range(self.n_classes_):
                coef, intercept = self._fit_binary(
                    X, y_encoded[:, i], self.lambda_bias, self.lambda_variance
                )
                coefs.append(coef)
                intercepts.append(intercept)
            
            self.coef_ = np.array(coefs)
            self.intercept_ = np.array(intercepts)
        
        return self
    
    def _optimize_parameters(self, X, y):
        """Optimize lambda_bias and lambda_variance via cross-validation."""
        from sklearn.model_selection import cross_val_score
        
        best_score = -np.inf
        best_params = {'lambda_bias': self.lambda_bias, 
                       'lambda_variance': self.lambda_variance}
        
        # Create parameter grid
        if 'lambda_bias' not in self.param_grid:
            lambda_bias_values = [self.lambda_bias]
        else:
            lambda_bias_values = self.param_grid['lambda_bias']
        
        if 'lambda_variance' not in self.param_grid:
            lambda_variance_values = [self.lambda_variance]
        else:
            lambda_variance_values = self.param_grid['lambda_variance']
        
        # Grid search
        for lb in lambda_bias_values:
            for lv in lambda_variance_values:
                # Create temporary model
                temp_model = BiasVarianceRidgeClassifier(
                    lambda_bias=lb,
                    lambda_variance=lv,
                    cv=None,
                    param_grid=None,
                    fit_intercept=self.fit_intercept
                )
                
                # Compute cross-validation score
                scores = cross_val_score(temp_model, X, y, cv=self.cv, 
                                        scoring='accuracy')
                mean_score = np.mean(scores)
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = {'lambda_bias': lb, 'lambda_variance': lv}
        
        # Update parameters
        self.lambda_bias = best_params['lambda_bias']
        self.lambda_variance = best_params['lambda_variance']
        self.best_score_ = best_score
    
    def decision_function(self, X):
        """
        Compute decision function.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        decision : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values
        """
        check_is_fitted(self)
        X = check_array(X)
        
        scores = X @ self.coef_.T + self.intercept_
        
        if self.n_classes_ == 2:
            return scores.ravel()
        return scores
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        check_is_fitted(self)
        X = check_array(X)
        
        scores = self.decision_function(X)
        
        if self.n_classes_ == 2:
            indices = (scores > 0).astype(int)
        else:
            indices = np.argmax(scores, axis=1)
        
        return self.classes_[indices]
    
    def predict_proba(self, X):
        """
        Predict class probabilities using softmax.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self)
        X = check_array(X)
        
        scores = self.decision_function(X)
        
        if self.n_classes_ == 2:
            # Binary case: use sigmoid
            scores = scores.reshape(-1, 1)
            proba_pos = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - proba_pos, proba_pos])
        else:
            # Multi-class: use softmax
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels
            
        Returns
        -------
        score : float
            Mean accuracy
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))