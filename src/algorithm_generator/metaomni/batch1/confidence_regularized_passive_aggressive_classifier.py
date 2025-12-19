import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class ConfidenceRegularizedPassiveAggressiveClassifier(BaseEstimator, ClassifierMixin):
    """
    Passive-Aggressive Classifier with L1-regularized updates that shrink towards
    simpler models when prediction margin exceeds confidence threshold.
    
    This classifier combines passive-aggressive learning with confidence-based
    regularization. When the prediction margin is large (high confidence), L1
    regularization is applied to shrink weights towards zero, promoting sparsity.
    Supports both binary and multi-class classification using one-vs-rest strategy.
    
    Parameters
    ----------
    C : float, default=1.0
        Maximum step size (regularization parameter). Higher values allow
        larger updates.
    
    confidence_threshold : float, default=1.0
        Margin threshold above which L1 regularization is applied.
        
    l1_strength : float, default=0.01
        Strength of L1 regularization applied when margin exceeds threshold.
        
    max_iter : int, default=1000
        Maximum number of passes over the training data.
        
    tol : float, default=1e-3
        Stopping criterion. Training stops when loss improvement is below this value.
        
    random_state : int, default=None
        Random seed for shuffling data.
        
    Attributes
    ----------
    coef_ : ndarray of shape (n_classes, n_features) or (n_features,)
        Weights assigned to the features.
        
    intercept_ : ndarray of shape (n_classes,) or float
        Intercept term.
        
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    """
    
    def __init__(self, C=1.0, confidence_threshold=1.0, l1_strength=0.01,
                 max_iter=1000, tol=1e-3, random_state=None):
        self.C = C
        self.confidence_threshold = confidence_threshold
        self.l1_strength = l1_strength
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
    
    def _soft_threshold(self, w, threshold):
        """Apply soft thresholding operator for L1 regularization."""
        return np.sign(w) * np.maximum(np.abs(w) - threshold, 0)
    
    def _hinge_loss(self, X, y, coef, intercept):
        """Compute average hinge loss."""
        margins = y * (np.dot(X, coef) + intercept)
        losses = np.maximum(0, 1 - margins)
        return np.mean(losses)
    
    def _fit_binary(self, X_train, y_binary):
        """
        Fit a single binary classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
            
        y_binary : array-like of shape (n_samples,)
            Binary target values (-1 or 1).
            
        Returns
        -------
        coef : ndarray of shape (n_features,)
            Fitted coefficients.
            
        intercept : float
            Fitted intercept.
        """
        n_samples, n_features = X_train.shape
        coef = np.zeros(n_features)
        intercept = 0.0
        
        # Set random state
        rng = np.random.RandomState(self.random_state)
        
        # Training loop
        prev_loss = float('inf')
        
        for iteration in range(self.max_iter):
            # Shuffle data
            indices = rng.permutation(n_samples)
            
            # Online updates
            for idx in indices:
                x_i = X_train[idx]
                y_i = y_binary[idx]
                
                # Compute margin
                margin = y_i * (np.dot(coef, x_i) + intercept)
                
                # Compute loss
                loss = max(0, 1 - margin)
                
                # Passive-Aggressive update when loss > 0
                if loss > 0:
                    # Compute step size (PA-I variant)
                    norm_sq = np.dot(x_i, x_i)
                    tau = min(self.C, loss / (norm_sq + 1e-10))
                    
                    # Update weights and intercept
                    coef += tau * y_i * x_i
                    intercept += tau * y_i
                
                # Apply L1 regularization when margin exceeds confidence threshold
                if margin > self.confidence_threshold:
                    # Soft thresholding for L1 regularization
                    coef = self._soft_threshold(coef, self.l1_strength)
            
            # Check convergence
            current_loss = self._hinge_loss(X_train, y_binary, coef, intercept)
            
            if abs(prev_loss - current_loss) < self.tol:
                break
            
            prev_loss = current_loss
        
        return coef, intercept
    
    def fit(self, X_train, y_train):
        """
        Fit the Confidence Regularized Passive-Aggressive classifier.
        
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
        n_classes = len(self.classes_)
        n_features = X_train.shape[1]
        
        if n_classes == 2:
            # Binary classification
            y_binary = np.where(y_train == self.classes_[0], -1, 1)
            self.coef_, self.intercept_ = self._fit_binary(X_train, y_binary)
            self.coef_ = self.coef_.reshape(1, -1)
            self.intercept_ = np.array([self.intercept_])
            
        else:
            # Multi-class classification using one-vs-rest
            self.coef_ = np.zeros((n_classes, n_features))
            self.intercept_ = np.zeros(n_classes)
            
            for i, class_label in enumerate(self.classes_):
                # Create binary labels: +1 for current class, -1 for others
                y_binary = np.where(y_train == class_label, 1, -1)
                
                # Fit binary classifier
                coef, intercept = self._fit_binary(X_train, y_binary)
                
                self.coef_[i] = coef
                self.intercept_[i] = intercept
        
        return self
    
    def decision_function(self, X_test):
        """
        Compute the decision function.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns
        -------
        decision : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values.
        """
        check_is_fitted(self, ['coef_', 'intercept_'])
        X_test = check_array(X_test)
        
        scores = np.dot(X_test, self.coef_.T) + self.intercept_
        
        if len(self.classes_) == 2:
            # Return 1D array for binary classification
            return scores.ravel()
        else:
            return scores
    
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
        check_is_fitted(self, ['coef_', 'intercept_'])
        X_test = check_array(X_test)
        
        # Get decision function values
        scores = self.decision_function(X_test)
        
        if len(self.classes_) == 2:
            # Binary classification
            y_pred = np.where(scores >= 0, self.classes_[1], self.classes_[0])
        else:
            # Multi-class classification: choose class with highest score
            indices = np.argmax(scores, axis=1)
            y_pred = self.classes_[indices]
        
        return y_pred
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'C': self.C,
            'confidence_threshold': self.confidence_threshold,
            'l1_strength': self.l1_strength,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self