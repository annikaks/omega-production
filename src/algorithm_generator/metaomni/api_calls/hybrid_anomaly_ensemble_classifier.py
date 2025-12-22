import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy import sparse
import warnings


class HybridAnomalyEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    A Hybrid Ensemble Classifier combining Gradient Boosting with Isolation Forest
    for feature-level anomaly detection, using soft-voting and custom scaling.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting stages for Gradient Boosting.
    
    n_trees : int, default=100
        Number of isolation trees for anomaly detection.
    
    max_samples : int or float, default='auto'
        Number of samples to draw for each isolation tree.
    
    contamination : float, default=0.1
        Expected proportion of anomalies in the dataset.
    
    learning_rate : float, default=0.1
        Learning rate for Gradient Boosting.
    
    max_depth : int, default=3
        Maximum depth of Gradient Boosting trees.
    
    anomaly_weight : float, default=0.3
        Weight given to anomaly scores in soft voting (0 to 1).
    
    scaling_method : str, default='robust'
        Internal scaling method: 'robust', 'standard', or 'minmax'.
    
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_estimators=100, n_trees=100, max_samples='auto',
                 contamination=0.1, learning_rate=0.1, max_depth=3,
                 anomaly_weight=0.3, scaling_method='robust', random_state=None):
        self.n_estimators = n_estimators
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.contamination = contamination
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.anomaly_weight = anomaly_weight
        self.scaling_method = scaling_method
        self.random_state = random_state
    
    def _custom_scale(self, X, fit=False):
        """Custom internal scaling method."""
        if sparse.issparse(X):
            X = X.toarray()
        
        if fit:
            if self.scaling_method == 'robust':
                self.median_ = np.median(X, axis=0)
                q75, q25 = np.percentile(X, [75, 25], axis=0)
                self.iqr_ = q75 - q25
                self.iqr_[self.iqr_ == 0] = 1.0
                return (X - self.median_) / self.iqr_
            
            elif self.scaling_method == 'standard':
                self.mean_ = np.mean(X, axis=0)
                self.std_ = np.std(X, axis=0)
                self.std_[self.std_ == 0] = 1.0
                return (X - self.mean_) / self.std_
            
            elif self.scaling_method == 'minmax':
                self.min_ = np.min(X, axis=0)
                self.max_ = np.max(X, axis=0)
                self.range_ = self.max_ - self.min_
                self.range_[self.range_ == 0] = 1.0
                return (X - self.min_) / self.range_
            
            else:
                return X
        else:
            if self.scaling_method == 'robust':
                return (X - self.median_) / self.iqr_
            elif self.scaling_method == 'standard':
                return (X - self.mean_) / self.std_
            elif self.scaling_method == 'minmax':
                return (X - self.min_) / self.range_
            else:
                return X
    
    def _build_isolation_tree(self, X, max_depth=10):
        """Build a single isolation tree."""
        n_samples, n_features = X.shape
        
        if n_samples <= 1 or max_depth == 0:
            return {'type': 'leaf', 'size': n_samples}
        
        # Randomly select feature and split value
        feature_idx = self.rng_.randint(0, n_features)
        feature_values = X[:, feature_idx]
        
        if len(np.unique(feature_values)) == 1:
            return {'type': 'leaf', 'size': n_samples}
        
        min_val, max_val = np.min(feature_values), np.max(feature_values)
        if min_val == max_val:
            return {'type': 'leaf', 'size': n_samples}
        
        split_value = self.rng_.uniform(min_val, max_val)
        
        left_mask = feature_values < split_value
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return {'type': 'leaf', 'size': n_samples}
        
        return {
            'type': 'node',
            'feature': feature_idx,
            'split': split_value,
            'left': self._build_isolation_tree(X[left_mask], max_depth - 1),
            'right': self._build_isolation_tree(X[right_mask], max_depth - 1)
        }
    
    def _path_length(self, x, tree, current_depth=0):
        """Calculate path length for a sample in an isolation tree."""
        if tree['type'] == 'leaf':
            size = tree['size']
            if size <= 1:
                return current_depth
            # Average path length of unsuccessful search in BST
            return current_depth + self._c(size)
        
        if x[tree['feature']] < tree['split']:
            return self._path_length(x, tree['left'], current_depth + 1)
        else:
            return self._path_length(x, tree['right'], current_depth + 1)
    
    def _c(self, n):
        """Average path length of unsuccessful search in BST."""
        if n <= 1:
            return 0
        return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n
    
    def _compute_anomaly_scores(self, X):
        """Compute anomaly scores using isolation forest."""
        n_samples = X.shape[0]
        path_lengths = np.zeros((n_samples, self.n_trees))
        
        for i, tree in enumerate(self.isolation_trees_):
            for j in range(n_samples):
                path_lengths[j, i] = self._path_length(X[j], tree)
        
        avg_path_lengths = np.mean(path_lengths, axis=1)
        
        # Normalize to anomaly scores (higher = more anomalous)
        if self.max_samples_ == 'auto':
            c = self._c(min(256, X.shape[0]))
        else:
            c = self._c(self.max_samples_)
        
        if c == 0:
            c = 1.0
        
        anomaly_scores = 2 ** (-avg_path_lengths / c)
        return anomaly_scores
    
    def fit(self, X, y):
        """
        Fit the hybrid ensemble classifier.
        
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
        X, y = check_X_y(X, y, accept_sparse=True)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Initialize random state
        self.rng_ = np.random.RandomState(self.random_state)
        
        # Convert sparse to dense if needed for scaling
        if sparse.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X.copy()
        
        # Apply custom scaling
        X_scaled = self._custom_scale(X_dense, fit=True)
        
        # Determine max_samples for isolation forest
        n_samples = X_scaled.shape[0]
        if self.max_samples == 'auto':
            self.max_samples_ = min(256, n_samples)
        elif isinstance(self.max_samples, float):
            self.max_samples_ = int(self.max_samples * n_samples)
        else:
            self.max_samples_ = min(self.max_samples, n_samples)
        
        # Build isolation forest
        self.isolation_trees_ = []
        max_tree_depth = int(np.ceil(np.log2(max(self.max_samples_, 2))))
        
        for _ in range(self.n_trees):
            # Sample data for this tree
            if self.max_samples_ < n_samples:
                sample_indices = self.rng_.choice(n_samples, 
                                                 size=self.max_samples_, 
                                                 replace=False)
            else:
                sample_indices = np.arange(n_samples)
            
            X_sample = X_scaled[sample_indices]
            
            # Build tree
            tree = self._build_isolation_tree(X_sample, max_depth=max_tree_depth)
            self.isolation_trees_.append(tree)
        
        # Compute anomaly scores for training data
        anomaly_scores = self._compute_anomaly_scores(X_scaled)
        self.anomaly_threshold_ = np.percentile(anomaly_scores, 
                                                (1 - self.contamination) * 100)
        
        # Create augmented features with anomaly scores
        X_augmented = np.column_stack([X_scaled, anomaly_scores.reshape(-1, 1)])
        
        # Train Gradient Boosting Classifier
        self.gb_classifier_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.gb_classifier_.fit(X_augmented, y)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities using soft voting.
        
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
        X = check_array(X, accept_sparse=True)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        
        # Convert sparse to dense if needed
        if sparse.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X.copy()
        
        # Apply scaling
        X_scaled = self._custom_scale(X_dense, fit=False)
        
        # Compute anomaly scores
        anomaly_scores = self._compute_anomaly_scores(X_scaled)
        
        # Create augmented features
        X_augmented = np.column_stack([X_scaled, anomaly_scores.reshape(-1, 1)])
        
        # Get GB predictions
        gb_proba = self.gb_classifier_.predict_proba(X_augmented)
        
        # Compute anomaly-based probabilities
        # Higher anomaly score -> more uncertain -> closer to uniform distribution
        anomaly_proba = np.copy(gb_proba)
        
        # Adjust based on anomaly scores
        for i in range(X.shape[0]):
            if anomaly_scores[i] > self.anomaly_threshold_:
                # High anomaly: increase uncertainty
                anomaly_factor = min(1.0, (anomaly_scores[i] - self.anomaly_threshold_) / 
                                   (1.0 - self.anomaly_threshold_ + 1e-10))
                anomaly_proba[i] = (1 - anomaly_factor) * gb_proba[i] + \
                                  anomaly_factor * (np.ones(self.n_classes_) / self.n_classes_)
        
        # Soft voting: weighted combination
        final_proba = (1 - self.anomaly_weight) * gb_proba + \
                     self.anomaly_weight * anomaly_proba
        
        # Normalize
        row_sums = final_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        final_proba = final_proba / row_sums
        
        return final_proba
    
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
    
    def get_anomaly_scores(self, X):
        """
        Get anomaly scores for samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        scores : array of shape (n_samples,)
            Anomaly scores (higher = more anomalous).
        """
        check_is_fitted(self)
        X = check_array(X, accept_sparse=True)
        
        if sparse.issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X.copy()
        
        X_scaled = self._custom_scale(X_dense, fit=False)
        return self._compute_anomaly_scores(X_scaled)