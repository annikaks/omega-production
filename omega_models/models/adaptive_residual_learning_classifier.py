import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import entropy


class AdaptiveResidualLearningClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that dynamically adjusts learning rates based on residual distribution.
    
    Uses boosting with adaptive learning rates that increase in underfit regions
    (high bias) and decrease in high variance regions.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Maximum number of estimators to train.
    base_learning_rate : float, default=0.1
        Initial learning rate for boosting.
    max_depth : int, default=3
        Maximum depth of base decision trees.
    variance_threshold : float, default=0.5
        Threshold for detecting high variance regions.
    underfit_threshold : float, default=0.3
        Threshold for detecting underfit regions.
    min_learning_rate : float, default=0.01
        Minimum allowed learning rate.
    max_learning_rate : float, default=1.0
        Maximum allowed learning rate.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_estimators=100, base_learning_rate=0.1, max_depth=3,
                 variance_threshold=0.5, underfit_threshold=0.3,
                 min_learning_rate=0.01, max_learning_rate=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.base_learning_rate = base_learning_rate
        self.max_depth = max_depth
        self.variance_threshold = variance_threshold
        self.underfit_threshold = underfit_threshold
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.random_state = random_state
    
    def _compute_residual_statistics(self, residuals, predictions_proba):
        """Compute statistics about residual distribution."""
        # Variance of residuals
        residual_variance = np.var(residuals)
        
        # Mean absolute residual (bias indicator)
        mean_abs_residual = np.mean(np.abs(residuals))
        
        # Entropy of prediction distribution (uncertainty)
        pred_entropy = np.mean([entropy(p + 1e-10) for p in predictions_proba])
        
        # Local variance (per-sample variance estimation)
        local_variance = np.abs(residuals - np.mean(residuals))
        
        return {
            'variance': residual_variance,
            'mean_abs_residual': mean_abs_residual,
            'entropy': pred_entropy,
            'local_variance': local_variance
        }
    
    def _adjust_learning_rate(self, stats, iteration):
        """Dynamically adjust learning rate based on residual characteristics."""
        base_lr = self.base_learning_rate
        
        # Decay factor based on iteration
        decay = 1.0 / (1.0 + 0.01 * iteration)
        
        # Increase learning rate for underfit regions (high bias)
        if stats['mean_abs_residual'] > self.underfit_threshold:
            # High bias - increase learning rate
            adjustment = 1.0 + (stats['mean_abs_residual'] - self.underfit_threshold)
        # Decrease learning rate for high variance regions
        elif stats['variance'] > self.variance_threshold:
            # High variance - decrease learning rate
            adjustment = 1.0 / (1.0 + stats['variance'])
        else:
            # Balanced region
            adjustment = 1.0
        
        # Apply entropy-based adjustment (higher entropy = more uncertainty = lower LR)
        entropy_factor = 1.0 / (1.0 + stats['entropy'])
        
        # Combine adjustments
        new_lr = base_lr * adjustment * entropy_factor * decay
        
        # Clip to valid range
        new_lr = np.clip(new_lr, self.min_learning_rate, self.max_learning_rate)
        
        return new_lr
    
    def _compute_sample_weights(self, residuals, stats):
        """Compute sample weights based on local residual characteristics."""
        # Higher weight for samples with high residuals (misclassified)
        weights = np.abs(residuals)
        
        # Adjust based on local variance
        local_var = stats['local_variance']
        
        # Reduce weight in high variance regions to prevent overfitting
        if len(local_var) > 0:
            percentile_75 = np.percentile(local_var, 75)
            variance_mask = local_var > percentile_75
            weights[variance_mask] *= 0.5
        
        # Increase weight in underfit regions
        underfit_mask = np.abs(residuals) > self.underfit_threshold
        weights[underfit_mask] *= 1.5
        
        # Normalize
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            weights = weights / weights_sum * len(weights)
        else:
            weights = np.ones_like(weights)
        
        weights = np.clip(weights, 0.1, 10.0)
        
        return weights
    
    def fit(self, X, y):
        """
        Fit the adaptive residual learning classifier.
        
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
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        # Initialize storage for estimators and learning rates
        self.estimators_ = []
        self.learning_rates_ = []
        
        # Convert labels to {-1, 1} for binary or use one-vs-rest for multiclass
        if self.n_classes_ == 2:
            y_encoded = np.where(y == self.classes_[0], -1, 1)
            self._fit_binary(X, y_encoded)
        else:
            self._fit_multiclass(X, y)
        
        return self
    
    def _fit_binary(self, X, y):
        """Fit binary classification with adaptive learning rates."""
        # Ensure lists are initialized
        if not hasattr(self, 'estimators_'):
            self.estimators_ = []
        if not hasattr(self, 'learning_rates_'):
            self.learning_rates_ = []
            
        n_samples = X.shape[0]
        
        # Initialize predictions
        F = np.zeros(n_samples)
        
        for i in range(self.n_estimators):
            # Compute residuals (negative gradient for exponential loss)
            predictions_proba = 1.0 / (1.0 + np.exp(-2 * F))
            predictions_proba = np.column_stack([1 - predictions_proba, predictions_proba])
            residuals = y - np.tanh(F)
            
            # Compute residual statistics
            stats = self._compute_residual_statistics(residuals, predictions_proba)
            
            # Adjust learning rate
            learning_rate = self._adjust_learning_rate(stats, i)
            self.learning_rates_.append(learning_rate)
            
            # Compute sample weights
            sample_weights = self._compute_sample_weights(residuals, stats)
            
            # Fit weak learner on residuals
            estimator = DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            
            # Use residuals as pseudo-responses
            y_pseudo = np.sign(residuals)
            # Handle zero residuals
            y_pseudo[y_pseudo == 0] = 1
            
            estimator.fit(X, y_pseudo, sample_weight=sample_weights)
            
            # Update predictions
            predictions = estimator.predict(X)
            F += learning_rate * predictions
            
            self.estimators_.append(estimator)
            
            # Early stopping if residuals are small
            if np.mean(np.abs(residuals)) < 0.01:
                break
    
    def _fit_multiclass(self, X, y):
        """Fit multiclass classification using one-vs-rest strategy."""
        self.binary_classifiers_ = []
        
        for class_label in self.classes_:
            y_binary = np.where(y == class_label, 1, -1)
            
            # Create a new classifier instance for this binary problem
            classifier = AdaptiveResidualLearningClassifier(
                n_estimators=self.n_estimators,
                base_learning_rate=self.base_learning_rate,
                max_depth=self.max_depth,
                variance_threshold=self.variance_threshold,
                underfit_threshold=self.underfit_threshold,
                min_learning_rate=self.min_learning_rate,
                max_learning_rate=self.max_learning_rate,
                random_state=self.random_state
            )
            
            # Initialize the lists before calling _fit_binary
            classifier.estimators_ = []
            classifier.learning_rates_ = []
            classifier.classes_ = np.array([-1, 1])
            classifier.n_classes_ = 2
            
            # Fit the binary classifier
            classifier._fit_binary(X, y_binary)
            self.binary_classifiers_.append(classifier)
    
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
        
        if self.n_classes_ == 2:
            F = np.zeros(X.shape[0])
            
            for estimator, lr in zip(self.estimators_, self.learning_rates_):
                predictions = estimator.predict(X)
                F += lr * predictions
            
            proba_positive = 1.0 / (1.0 + np.exp(-2 * F))
            proba_positive = np.clip(proba_positive, 1e-10, 1 - 1e-10)
            return np.column_stack([1 - proba_positive, proba_positive])
        else:
            # Multiclass: aggregate predictions from binary classifiers
            scores = np.zeros((X.shape[0], self.n_classes_))
            
            for i, classifier in enumerate(self.binary_classifiers_):
                F = np.zeros(X.shape[0])
                for estimator, lr in zip(classifier.estimators_, classifier.learning_rates_):
                    predictions = estimator.predict(X)
                    F += lr * predictions
                scores[:, i] = F
            
            # Softmax
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
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