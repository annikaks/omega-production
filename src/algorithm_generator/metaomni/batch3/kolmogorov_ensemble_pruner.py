import numpy as np
import zlib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from typing import List, Any


class KolmogorovEnsemblePruner(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier that uses Kolmogorov complexity estimation to prune
    redundant ensemble members that don't contribute unique pattern descriptions.
    
    The classifier builds an ensemble of base estimators and prunes members whose
    prediction patterns can be compressed well given other members' patterns,
    indicating redundancy.
    
    Parameters
    ----------
    n_estimators : int, default=50
        Number of base estimators to initially train.
    
    complexity_threshold : float, default=0.85
        Threshold for normalized compression distance. Members with NCD below
        this threshold relative to existing ensemble are considered redundant.
    
    min_estimators : int, default=5
        Minimum number of estimators to retain after pruning.
    
    base_estimator : estimator object, default=None
        The base estimator to use. If None, uses DecisionTreeClassifier.
    
    random_state : int, default=None
        Random state for reproducibility.
    
    Attributes
    ----------
    estimators_ : list
        The pruned collection of fitted base estimators.
    
    classes_ : ndarray of shape (n_classes,)
        The class labels.
    """
    
    def __init__(
        self,
        n_estimators=50,
        complexity_threshold=0.85,
        min_estimators=5,
        base_estimator=None,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.complexity_threshold = complexity_threshold
        self.min_estimators = min_estimators
        self.base_estimator = base_estimator
        self.random_state = random_state
    
    def _kolmogorov_complexity(self, data: bytes) -> int:
        """
        Estimate Kolmogorov complexity using compression length.
        
        Parameters
        ----------
        data : bytes
            Data to estimate complexity for.
        
        Returns
        -------
        int
            Compressed length as proxy for Kolmogorov complexity.
        """
        return len(zlib.compress(data, level=9))
    
    def _normalized_compression_distance(self, x: bytes, y: bytes) -> float:
        """
        Calculate Normalized Compression Distance between two byte sequences.
        
        NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
        
        Parameters
        ----------
        x, y : bytes
            Byte sequences to compare.
        
        Returns
        -------
        float
            NCD value between 0 and 1+epsilon.
        """
        cx = self._kolmogorov_complexity(x)
        cy = self._kolmogorov_complexity(y)
        cxy = self._kolmogorov_complexity(x + y)
        
        if max(cx, cy) == 0:
            return 0.0
        
        return (cxy - min(cx, cy)) / max(cx, cy)
    
    def _predictions_to_bytes(self, predictions: np.ndarray) -> bytes:
        """Convert prediction array to bytes for compression."""
        return predictions.tobytes()
    
    def _is_unique_pattern(
        self,
        candidate_preds: np.ndarray,
        ensemble_preds_list: List[np.ndarray]
    ) -> bool:
        """
        Determine if candidate provides unique patterns compared to ensemble.
        
        Parameters
        ----------
        candidate_preds : ndarray
            Predictions from candidate estimator.
        
        ensemble_preds_list : list of ndarray
            List of predictions from current ensemble members.
        
        Returns
        -------
        bool
            True if candidate provides unique patterns.
        """
        if len(ensemble_preds_list) == 0:
            return True
        
        candidate_bytes = self._predictions_to_bytes(candidate_preds)
        
        # Compare with each ensemble member
        min_ncd = float('inf')
        for ensemble_preds in ensemble_preds_list:
            ensemble_bytes = self._predictions_to_bytes(ensemble_preds)
            ncd = self._normalized_compression_distance(candidate_bytes, ensemble_bytes)
            min_ncd = min(min_ncd, ncd)
        
        # If minimum NCD is above threshold, patterns are sufficiently unique
        return min_ncd >= self.complexity_threshold
    
    def fit(self, X, y):
        """
        Build ensemble and prune redundant members using Kolmogorov complexity.
        
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
        self.classes_ = unique_labels(y)
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Train initial ensemble
        initial_estimators = []
        for i in range(self.n_estimators):
            if self.base_estimator is None:
                estimator = DecisionTreeClassifier(
                    max_depth=10,
                    min_samples_split=5,
                    random_state=rng.randint(0, 10000)
                )
            else:
                estimator = self.base_estimator.__class__(**self.base_estimator.get_params())
            
            # Bootstrap sample
            indices = rng.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            estimator.fit(X_bootstrap, y_bootstrap)
            initial_estimators.append(estimator)
        
        # Prune ensemble using Kolmogorov complexity
        self.estimators_ = []
        ensemble_predictions = []
        
        # Get predictions from all estimators on training data
        all_predictions = [est.predict(X) for est in initial_estimators]
        
        # Greedy selection: add estimators with unique patterns
        for i, estimator in enumerate(initial_estimators):
            if len(self.estimators_) < self.min_estimators:
                # Always keep minimum number of estimators
                self.estimators_.append(estimator)
                ensemble_predictions.append(all_predictions[i])
            else:
                # Check if this estimator provides unique patterns
                if self._is_unique_pattern(all_predictions[i], ensemble_predictions):
                    self.estimators_.append(estimator)
                    ensemble_predictions.append(all_predictions[i])
        
        # Ensure we have at least min_estimators
        if len(self.estimators_) < self.min_estimators:
            for i, estimator in enumerate(initial_estimators):
                if estimator not in self.estimators_:
                    self.estimators_.append(estimator)
                    if len(self.estimators_) >= self.min_estimators:
                        break
        
        self.n_estimators_ = len(self.estimators_)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels using the pruned ensemble.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ['estimators_', 'classes_'])
        X = check_array(X)
        
        # Collect predictions from all ensemble members
        predictions = np.array([est.predict(X) for est in self.estimators_])
        
        # Majority voting
        y_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the pruned ensemble.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self, ['estimators_', 'classes_'])
        X = check_array(X)
        
        # Average probabilities from all ensemble members
        all_proba = np.array([est.predict_proba(X) for est in self.estimators_])
        avg_proba = np.mean(all_proba, axis=0)
        
        return avg_proba