import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler


class AdaptiveAggressivenessClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Aggressiveness Classifier that dynamically adjusts model complexity
    based on prediction confidence to balance bias-variance tradeoff.
    
    The classifier uses an ensemble of models with varying complexity levels.
    For high-confidence predictions, it uses simpler models (lower variance).
    For low-confidence predictions, it uses more complex models (lower bias).
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the random forest (complex model).
    
    max_depth_simple : int, default=3
        Maximum depth for simple model (low aggressiveness).
    
    max_depth_complex : int, default=None
        Maximum depth for complex model (high aggressiveness).
    
    confidence_threshold_low : float, default=0.6
        Lower confidence threshold for model selection.
    
    confidence_threshold_high : float, default=0.8
        Upper confidence threshold for model selection.
    
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(
        self,
        n_estimators=100,
        max_depth_simple=3,
        max_depth_complex=None,
        confidence_threshold_low=0.6,
        confidence_threshold_high=0.8,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_depth_simple = max_depth_simple
        self.max_depth_complex = max_depth_complex
        self.confidence_threshold_low = confidence_threshold_low
        self.confidence_threshold_high = confidence_threshold_high
        self.random_state = random_state
    
    def fit(self, X_train, y_train):
        """
        Fit the adaptive aggressiveness classifier.
        
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
        self.classes_ = unique_labels(y_train)
        self.n_features_in_ = X_train.shape[1]
        
        # Initialize scaler
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_train)
        
        # Train simple model (low aggressiveness, high bias, low variance)
        self.simple_model_ = LogisticRegression(
            C=0.1,  # High regularization
            max_iter=1000,
            random_state=self.random_state
        )
        self.simple_model_.fit(X_scaled, y_train)
        
        # Train medium model (moderate aggressiveness)
        self.medium_model_ = RandomForestClassifier(
            n_estimators=self.n_estimators // 2,
            max_depth=self.max_depth_simple * 2,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_state
        )
        self.medium_model_.fit(X_train, y_train)
        
        # Train complex model (high aggressiveness, low bias, high variance)
        self.complex_model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth_complex,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=self.random_state
        )
        self.complex_model_.fit(X_train, y_train)
        
        # Train confidence estimator (meta-model)
        self.confidence_model_ = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state
        )
        self.confidence_model_.fit(X_scaled, y_train)
        
        return self
    
    def _compute_confidence(self, X):
        """
        Compute prediction confidence for each sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        
        Returns
        -------
        confidence : array-like of shape (n_samples,)
            Confidence scores for each sample.
        """
        X_scaled = self.scaler_.transform(X)
        
        # Get probability predictions from confidence model
        proba = self.confidence_model_.predict_proba(X_scaled)
        
        # Confidence is the maximum probability
        confidence = np.max(proba, axis=1)
        
        return confidence
    
    def _get_aggressiveness_weights(self, confidence):
        """
        Compute model weights based on confidence levels.
        
        Parameters
        ----------
        confidence : array-like of shape (n_samples,)
            Confidence scores.
        
        Returns
        -------
        weights : dict
            Dictionary with weights for each model.
        """
        n_samples = len(confidence)
        weights = {
            'simple': np.zeros(n_samples),
            'medium': np.zeros(n_samples),
            'complex': np.zeros(n_samples)
        }
        
        for i, conf in enumerate(confidence):
            if conf >= self.confidence_threshold_high:
                # High confidence: use simple model (low aggressiveness)
                weights['simple'][i] = 1.0
            elif conf <= self.confidence_threshold_low:
                # Low confidence: use complex model (high aggressiveness)
                weights['complex'][i] = 1.0
            else:
                # Medium confidence: interpolate between medium and simple
                alpha = (conf - self.confidence_threshold_low) / (
                    self.confidence_threshold_high - self.confidence_threshold_low
                )
                weights['simple'][i] = alpha
                weights['medium'][i] = 1.0 - alpha
        
        return weights
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        # Compute confidence for each sample
        confidence = self._compute_confidence(X_test)
        
        # Get aggressiveness weights
        weights = self._get_aggressiveness_weights(confidence)
        
        # Get predictions from all models
        X_scaled = self.scaler_.transform(X_test)
        proba_simple = self.simple_model_.predict_proba(X_scaled)
        proba_medium = self.medium_model_.predict_proba(X_test)
        proba_complex = self.complex_model_.predict_proba(X_test)
        
        # Weighted combination
        proba = np.zeros_like(proba_simple)
        for i in range(len(X_test)):
            proba[i] = (
                weights['simple'][i] * proba_simple[i] +
                weights['medium'][i] * proba_medium[i] +
                weights['complex'][i] * proba_complex[i]
            )
            # Normalize to ensure valid probabilities
            proba[i] /= proba[i].sum()
        
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
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def get_aggressiveness_profile(self, X_test):
        """
        Get the aggressiveness profile for test samples.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        profile : dict
            Dictionary containing confidence scores and model weights.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        confidence = self._compute_confidence(X_test)
        weights = self._get_aggressiveness_weights(confidence)
        
        return {
            'confidence': confidence,
            'simple_weight': weights['simple'],
            'medium_weight': weights['medium'],
            'complex_weight': weights['complex']
        }