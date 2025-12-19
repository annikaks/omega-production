import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize


class AdditiveRidgeSegmentClassifier(BaseEstimator, ClassifierMixin):
    """
    Piecewise-linear additive classifier with region-specific ridge parameters.
    
    Uses decision trees to partition feature space into regions, then fits
    additive linear models with region-specific ridge regularization in each segment.
    
    Parameters
    ----------
    n_segments : int, default=5
        Number of segments per feature for piecewise approximation
    alpha : float, default=1.0
        Base ridge regularization parameter
    adaptive_alpha : bool, default=True
        Whether to use region-specific ridge parameters
    max_depth : int, default=3
        Maximum depth of trees for region partitioning
    random_state : int, default=None
        Random state for reproducibility
    """
    
    def __init__(self, n_segments=5, alpha=1.0, adaptive_alpha=True, 
                 max_depth=3, random_state=None):
        self.n_segments = n_segments
        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha
        self.max_depth = max_depth
        self.random_state = random_state
        
    def _partition_space(self, X, y):
        """Partition feature space using decision trees."""
        self.trees_ = []
        self.feature_splits_ = []
        
        for j in range(X.shape[1]):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=max(10, X.shape[0] // (self.n_segments * 2)),
                random_state=self.random_state
            )
            tree.fit(X[:, j:j+1], y)
            self.trees_.append(tree)
            
            # Extract split points
            splits = self._extract_splits(tree, X[:, j])
            self.feature_splits_.append(splits)
    
    def _extract_splits(self, tree, feature_values):
        """Extract split points from decision tree."""
        splits = []
        
        def traverse(node_id):
            if tree.tree_.feature[node_id] != -2:  # Not a leaf
                threshold = tree.tree_.threshold[node_id]
                splits.append(threshold)
                traverse(tree.tree_.children_left[node_id])
                traverse(tree.tree_.children_right[node_id])
        
        traverse(0)
        splits = sorted(list(set(splits)))
        
        # Add boundary points
        min_val, max_val = feature_values.min(), feature_values.max()
        splits = [min_val - 1e-6] + splits + [max_val + 1e-6]
        
        return np.array(splits)
    
    def _get_region_indicators(self, X):
        """Get region indicator matrix for all samples."""
        n_samples, n_features = X.shape
        region_features = []
        self.region_info_ = []
        
        for j in range(n_features):
            splits = self.feature_splits_[j]
            n_regions = len(splits) - 1
            
            for r in range(n_regions):
                # Create indicator for region
                in_region = (X[:, j] >= splits[r]) & (X[:, j] < splits[r + 1])
                
                # Linear component within region
                region_feature = np.zeros(n_samples)
                region_feature[in_region] = X[in_region, j]
                
                region_features.append(region_feature)
                self.region_info_.append({
                    'feature': j,
                    'region': r,
                    'bounds': (splits[r], splits[r + 1])
                })
        
        return np.column_stack(region_features)
    
    def _compute_region_alphas(self, X_regions, y):
        """Compute region-specific ridge parameters."""
        n_regions = X_regions.shape[1]
        alphas = np.ones(n_regions) * self.alpha
        
        if self.adaptive_alpha:
            for r in range(n_regions):
                # Compute variance in region
                active = X_regions[:, r] != 0
                if active.sum() > 1:
                    var = np.var(X_regions[active, r])
                    # Adapt alpha based on local variance
                    alphas[r] = self.alpha * (1.0 + 1.0 / (1.0 + var))
        
        return alphas
    
    def fit(self, X_train, y_train):
        """
        Fit the additive ridge segment classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        # Handle classification labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y_train)
        self.classes_ = self.label_encoder_.classes_
        
        # Convert to binary for binary classification, use OvR for multiclass
        if len(self.classes_) == 2:
            self._fit_binary(X_train, y_encoded)
        else:
            self._fit_multiclass(X_train, y_encoded)
        
        return self
    
    def _fit_binary(self, X, y):
        """Fit binary classifier."""
        # Partition space
        self._partition_space(X, y)
        
        # Get region indicators
        X_regions = self._get_region_indicators(X)
        
        # Compute region-specific alphas
        self.alphas_ = self._compute_region_alphas(X_regions, y)
        
        # Fit ridge regression with region-specific regularization
        n_regions = X_regions.shape[1]
        
        # Add intercept
        X_aug = np.column_stack([np.ones(X.shape[0]), X_regions])
        
        # Ridge regression with diagonal regularization matrix
        Lambda = np.diag(np.concatenate([[0], self.alphas_]))  # No penalty on intercept
        
        # Closed form solution: (X^T X + Lambda)^-1 X^T y
        XtX = X_aug.T @ X_aug
        Xty = X_aug.T @ y
        
        self.coef_ = np.linalg.solve(XtX + Lambda, Xty)
        self.intercept_ = self.coef_[0]
        self.weights_ = self.coef_[1:]
    
    def _fit_multiclass(self, X, y):
        """Fit multiclass classifier using One-vs-Rest."""
        n_classes = len(self.classes_)
        self.binary_classifiers_ = []
        
        for c in range(n_classes):
            y_binary = (y == c).astype(int)
            clf = AdditiveRidgeSegmentClassifier(
                n_segments=self.n_segments,
                alpha=self.alpha,
                adaptive_alpha=self.adaptive_alpha,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            clf._fit_binary(X, y_binary)
            self.binary_classifiers_.append(clf)
    
    def _decision_function_binary(self, X):
        """Compute decision function for binary classification."""
        X = np.asarray(X)
        X_regions = self._get_region_indicators(X)
        X_aug = np.column_stack([np.ones(X.shape[0]), X_regions])
        return X_aug @ self.coef_
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities
        """
        X_test = np.asarray(X_test)
        
        if len(self.classes_) == 2:
            scores = self._decision_function_binary(X_test)
            proba = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - proba, proba])
        else:
            scores = np.column_stack([
                clf._decision_function_binary(X_test)
                for clf in self.binary_classifiers_
            ])
            # Softmax
            exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
            return exp_scores / exp_scores.sum(axis=1, keepdims=True)
    
    def predict(self, X_test):
        """
        Predict class labels.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X_test)
        return self.label_encoder_.inverse_transform(np.argmax(proba, axis=1))