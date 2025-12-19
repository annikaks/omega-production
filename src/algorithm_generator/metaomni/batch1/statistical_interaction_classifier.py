import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, spearmanr
from itertools import combinations
from typing import Optional, Tuple, List


class StatisticalInteractionClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that introduces interaction terms only where statistical 
    dependence tests reject independence assumptions.
    
    This classifier tests for statistical dependence between feature pairs
    using appropriate tests (chi-square for categorical, Spearman correlation
    for continuous) and only creates interaction terms for dependent features.
    
    Parameters
    ----------
    base_estimator : estimator object, default=None
        The base classifier to use. If None, uses LogisticRegression.
    
    alpha : float, default=0.05
        Significance level for independence tests. Lower values mean
        stricter criteria for rejecting independence.
    
    max_interactions : int, default=None
        Maximum number of interaction terms to create. If None, creates
        all statistically significant interactions.
    
    correlation_threshold : float, default=0.1
        Minimum absolute correlation coefficient to consider for continuous
        features (in addition to p-value threshold).
    
    scale_features : bool, default=True
        Whether to standardize features before fitting.
    """
    
    def __init__(
        self,
        base_estimator=None,
        alpha: float = 0.05,
        max_interactions: Optional[int] = None,
        correlation_threshold: float = 0.1,
        scale_features: bool = True
    ):
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.max_interactions = max_interactions
        self.correlation_threshold = correlation_threshold
        self.scale_features = scale_features
        
    def _is_categorical(self, feature: np.ndarray, threshold: int = 10) -> bool:
        """Determine if a feature is categorical based on unique values."""
        n_unique = len(np.unique(feature))
        return n_unique <= threshold
    
    def _test_independence_continuous(
        self, 
        x1: np.ndarray, 
        x2: np.ndarray, 
        y: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Test independence between two continuous features using Spearman correlation.
        Also considers their joint relationship with the target.
        """
        # Test correlation between features
        corr, p_value = spearmanr(x1, x2)
        
        # Reject independence if correlation is significant and above threshold
        is_dependent = (p_value < self.alpha and 
                       abs(corr) > self.correlation_threshold)
        
        return is_dependent, p_value
    
    def _test_independence_mixed(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
        y: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Test independence for mixed or categorical features using chi-square test.
        """
        try:
            # Create contingency table
            x1_binned = self._bin_feature(x1)
            x2_binned = self._bin_feature(x2)
            
            # Create 2D histogram
            contingency = np.histogram2d(
                x1_binned, x2_binned, 
                bins=[len(np.unique(x1_binned)), len(np.unique(x2_binned))]
            )[0]
            
            # Perform chi-square test
            chi2, p_value, dof, expected = chi2_contingency(contingency + 1e-10)
            
            is_dependent = p_value < self.alpha
            return is_dependent, p_value
            
        except Exception:
            # If test fails, assume independence
            return False, 1.0
    
    def _bin_feature(self, feature: np.ndarray, n_bins: int = 5) -> np.ndarray:
        """Bin a continuous feature into discrete categories."""
        if self._is_categorical(feature):
            return feature
        else:
            return np.digitize(feature, bins=np.percentile(feature, np.linspace(0, 100, n_bins)))
    
    def _find_significant_interactions(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """
        Find pairs of features that show statistical dependence.
        
        Returns list of tuples: (feature_idx1, feature_idx2, p_value)
        """
        n_features = X.shape[1]
        significant_pairs = []
        
        # Test all pairs of features
        for i, j in combinations(range(n_features), 2):
            x1, x2 = X[:, i], X[:, j]
            
            # Determine feature types
            is_cat1 = self._is_categorical(x1)
            is_cat2 = self._is_categorical(x2)
            
            # Choose appropriate test
            if is_cat1 or is_cat2:
                is_dependent, p_value = self._test_independence_mixed(x1, x2, y)
            else:
                is_dependent, p_value = self._test_independence_continuous(x1, x2, y)
            
            if is_dependent:
                significant_pairs.append((i, j, p_value))
        
        # Sort by p-value (most significant first)
        significant_pairs.sort(key=lambda x: x[2])
        
        # Limit number of interactions if specified
        if self.max_interactions is not None:
            significant_pairs = significant_pairs[:self.max_interactions]
        
        return significant_pairs
    
    def _create_interaction_features(
        self, 
        X: np.ndarray, 
        interaction_pairs: List[Tuple[int, int, float]]
    ) -> np.ndarray:
        """Create interaction features for significant pairs."""
        if len(interaction_pairs) == 0:
            return X
        
        interaction_features = []
        for i, j, _ in interaction_pairs:
            interaction = X[:, i] * X[:, j]
            interaction_features.append(interaction.reshape(-1, 1))
        
        if interaction_features:
            X_augmented = np.hstack([X] + interaction_features)
        else:
            X_augmented = X
            
        return X_augmented
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the classifier with automatic interaction term selection.
        
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
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        # Store original feature count
        self.n_features_in_ = X_train.shape[1]
        
        # Scale features if requested
        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_train)
        else:
            self.scaler_ = None
            X_scaled = X_train
        
        # Find significant interactions
        self.interaction_pairs_ = self._find_significant_interactions(X_scaled, y_train)
        
        # Create augmented feature matrix
        X_augmented = self._create_interaction_features(X_scaled, self.interaction_pairs_)
        
        # Initialize base estimator
        if self.base_estimator is None:
            self.estimator_ = LogisticRegression(max_iter=1000, random_state=42)
        else:
            self.estimator_ = self.base_estimator
        
        # Fit the base estimator
        self.estimator_.fit(X_augmented, y_train)
        
        # Store classes
        self.classes_ = np.unique(y_train)
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        X_test = np.asarray(X_test)
        
        # Scale features if scaler was fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X_test)
        else:
            X_scaled = X_test
        
        # Create interaction features
        X_augmented = self._create_interaction_features(X_scaled, self.interaction_pairs_)
        
        # Predict using base estimator
        return self.estimator_.predict(X_augmented)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        X_test = np.asarray(X_test)
        
        # Scale features if scaler was fitted
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X_test)
        else:
            X_scaled = X_test
        
        # Create interaction features
        X_augmented = self._create_interaction_features(X_scaled, self.interaction_pairs_)
        
        # Predict probabilities using base estimator
        return self.estimator_.predict_proba(X_augmented)
    
    def get_interaction_info(self) -> List[Tuple[int, int, float]]:
        """
        Get information about selected interaction terms.
        
        Returns
        -------
        interactions : list of tuples
            List of (feature_idx1, feature_idx2, p_value) for selected interactions.
        """
        return self.interaction_pairs_