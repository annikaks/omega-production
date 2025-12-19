import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import binned_statistic


class ResidualAwareHistogramBoostingClassifier(BaseEstimator, ClassifierMixin):
    """
    A gradient boosting classifier that uses residual-aware histogram construction.
    
    This classifier allocates more bins to regions where previous boosting iterations
    show high residual variance, allowing for more refined splits in difficult areas.
    Supports both binary and multi-class classification.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting iterations.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree.
    max_depth : int, default=3
        Maximum depth of the individual decision trees.
    n_bins : int, default=32
        Base number of bins for histogram construction.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    residual_percentile : float, default=75
        Percentile threshold for identifying high residual variance regions.
    adaptive_bin_ratio : float, default=2.0
        Ratio of bins to allocate to high-variance regions vs low-variance regions.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 n_bins=32, min_samples_leaf=1, residual_percentile=75,
                 adaptive_bin_ratio=2.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.min_samples_leaf = min_samples_leaf
        self.residual_percentile = residual_percentile
        self.adaptive_bin_ratio = adaptive_bin_ratio
        self.random_state = random_state
    
    def _create_adaptive_bins(self, X, residuals):
        """Create adaptive bin edges based on residual variance."""
        n_samples, n_features = X.shape
        bin_edges_list = []
        
        # Compute overall residual magnitude for each sample
        residual_magnitude = np.abs(residuals)
        
        # Handle edge case where all residuals are the same
        if np.std(residual_magnitude) < 1e-10:
            for feat_idx in range(n_features):
                feature_vals = X[:, feat_idx]
                quantiles = np.linspace(0, 1, self.n_bins + 1)
                edges = np.percentile(feature_vals, quantiles * 100)
                bin_edges_list.append(np.unique(edges))
            return bin_edges_list
        
        threshold = np.percentile(residual_magnitude, self.residual_percentile)
        high_residual_mask = residual_magnitude >= threshold
        
        for feat_idx in range(n_features):
            feature_vals = X[:, feat_idx]
            
            # Separate samples into high and low residual regions
            high_res_vals = feature_vals[high_residual_mask]
            low_res_vals = feature_vals[~high_residual_mask]
            
            # Allocate bins proportionally
            total_bins = self.n_bins
            if len(high_res_vals) > 0 and len(low_res_vals) > 0:
                high_res_bins = int(total_bins * self.adaptive_bin_ratio / (1 + self.adaptive_bin_ratio))
                low_res_bins = total_bins - high_res_bins
                
                # Ensure at least 2 bins for each region
                high_res_bins = max(2, high_res_bins)
                low_res_bins = max(2, low_res_bins)
                
                # Create quantile-based bins for each region
                high_quantiles = np.linspace(0, 100, high_res_bins + 1)
                low_quantiles = np.linspace(0, 100, low_res_bins + 1)
                
                high_edges = np.percentile(high_res_vals, high_quantiles)
                low_edges = np.percentile(low_res_vals, low_quantiles)
                
                # Combine and sort unique edges
                combined_edges = np.unique(np.concatenate([high_edges, low_edges]))
            else:
                # Fallback to uniform quantile bins
                quantiles = np.linspace(0, 100, total_bins + 1)
                combined_edges = np.percentile(feature_vals, quantiles)
                combined_edges = np.unique(combined_edges)
            
            bin_edges_list.append(combined_edges)
            
        return bin_edges_list
    
    def _discretize_features(self, X, bin_edges_list):
        """Discretize features using provided bin edges."""
        X_binned = np.zeros_like(X, dtype=np.float64)
        
        for feat_idx, bin_edges in enumerate(bin_edges_list):
            if len(bin_edges) > 1:
                X_binned[:, feat_idx] = np.digitize(X[:, feat_idx], bin_edges[1:-1])
            else:
                X_binned[:, feat_idx] = 0
            
        return X_binned
    
    def _softmax(self, x):
        """Numerically stable softmax function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _sigmoid(self, x):
        """Numerically stable sigmoid function."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-np.clip(x, -500, 500))),
            np.exp(np.clip(x, -500, 500)) / (1 + np.exp(np.clip(x, -500, 500)))
        )
    
    def fit(self, X, y):
        """
        Fit the residual-aware histogram boosting classifier.
        
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
        self.n_classes_ = len(self.classes_)
        
        # Map labels to indices
        self.label_map_ = {label: idx for idx, label in enumerate(self.classes_)}
        y_encoded = np.array([self.label_map_[label] for label in y])
        
        n_samples = X.shape[0]
        self.estimators_ = []
        self.bin_edges_history_ = []
        
        if self.n_classes_ == 2:
            # Binary classification
            y_binary = (y_encoded == 1).astype(np.float64)
            
            # Initial prediction (log-odds of positive class)
            pos_ratio = np.mean(y_binary)
            pos_ratio = np.clip(pos_ratio, 1e-10, 1 - 1e-10)
            self.init_pred_ = np.log(pos_ratio / (1 - pos_ratio))
            F = np.full(n_samples, self.init_pred_)
            
            # Boosting iterations
            for iteration in range(self.n_estimators):
                # Compute probabilities and residuals (negative gradient)
                probs = self._sigmoid(F)
                residuals = y_binary - probs
                
                # Create adaptive bins based on residual variance
                if iteration == 0:
                    # First iteration: use uniform bins
                    bin_edges_list = []
                    for feat_idx in range(X.shape[1]):
                        quantiles = np.linspace(0, 100, self.n_bins + 1)
                        edges = np.percentile(X[:, feat_idx], quantiles)
                        bin_edges_list.append(np.unique(edges))
                else:
                    # Subsequent iterations: use residual-aware bins
                    bin_edges_list = self._create_adaptive_bins(X, residuals)
                
                self.bin_edges_history_.append(bin_edges_list)
                
                # Discretize features
                X_binned = self._discretize_features(X, bin_edges_list)
                
                # Fit a decision tree regressor on binned features
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state
                )
                
                # Fit tree to predict residuals
                tree.fit(X_binned, residuals)
                
                # Predict residuals
                pred_residuals = tree.predict(X_binned)
                
                # Update predictions
                F += self.learning_rate * pred_residuals
                
                self.estimators_.append(tree)
        else:
            # Multi-class classification using one-vs-rest
            self.init_pred_ = np.zeros(self.n_classes_)
            for k in range(self.n_classes_):
                class_count = np.sum(y_encoded == k)
                if class_count > 0:
                    self.init_pred_[k] = np.log(class_count / n_samples + 1e-10)
            
            F = np.tile(self.init_pred_, (n_samples, 1))
            
            # Store estimators for each class
            self.estimators_ = [[] for _ in range(self.n_classes_)]
            self.bin_edges_history_ = [[] for _ in range(self.n_classes_)]
            
            # Boosting iterations
            for iteration in range(self.n_estimators):
                # Compute probabilities using softmax
                probs = self._softmax(F)
                
                # Train a tree for each class
                for k in range(self.n_classes_):
                    # Compute residuals for class k
                    y_k = (y_encoded == k).astype(np.float64)
                    residuals = y_k - probs[:, k]
                    
                    # Create adaptive bins
                    if iteration == 0:
                        bin_edges_list = []
                        for feat_idx in range(X.shape[1]):
                            quantiles = np.linspace(0, 100, self.n_bins + 1)
                            edges = np.percentile(X[:, feat_idx], quantiles)
                            bin_edges_list.append(np.unique(edges))
                    else:
                        bin_edges_list = self._create_adaptive_bins(X, residuals)
                    
                    self.bin_edges_history_[k].append(bin_edges_list)
                    
                    # Discretize features
                    X_binned = self._discretize_features(X, bin_edges_list)
                    
                    # Fit tree
                    tree = DecisionTreeRegressor(
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        random_state=self.random_state
                    )
                    tree.fit(X_binned, residuals)
                    
                    # Update predictions for class k
                    pred_residuals = tree.predict(X_binned)
                    F[:, k] += self.learning_rate * pred_residuals
                    
                    self.estimators_[k].append(tree)
        
        self.is_fitted_ = True
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
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        
        n_samples = X.shape[0]
        
        if self.n_classes_ == 2:
            # Binary classification
            F = np.full(n_samples, self.init_pred_)
            
            # Accumulate predictions from all estimators
            for iteration, (tree, bin_edges_list) in enumerate(
                zip(self.estimators_, self.bin_edges_history_)
            ):
                X_binned = self._discretize_features(X, bin_edges_list)
                pred_residuals = tree.predict(X_binned)
                F += self.learning_rate * pred_residuals
            
            # Convert to probabilities
            probs_positive = self._sigmoid(F)
            probs = np.column_stack([1 - probs_positive, probs_positive])
        else:
            # Multi-class classification
            F = np.tile(self.init_pred_, (n_samples, 1))
            
            # Accumulate predictions for each class
            for k in range(self.n_classes_):
                for iteration, (tree, bin_edges_list) in enumerate(
                    zip(self.estimators_[k], self.bin_edges_history_[k])
                ):
                    X_binned = self._discretize_features(X, bin_edges_list)
                    pred_residuals = tree.predict(X_binned)
                    F[:, k] += self.learning_rate * pred_residuals
            
            # Convert to probabilities using softmax
            probs = self._softmax(F)
        
        return probs
    
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