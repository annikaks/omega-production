import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class AdaptiveHistogramTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Decision tree classifier with dynamically adjusted histogram bin granularity.
    
    Uses fine-grained bins at shallow depths and progressively coarser bins
    at deeper levels to balance accuracy and computational efficiency.
    
    Parameters
    ----------
    max_depth : int, default=10
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node.
    initial_bins : int, default=256
        Number of bins at the root level.
    bin_decay_rate : float, default=0.7
        Rate at which bins decrease with depth (bins = initial_bins * decay_rate^depth).
    min_bins : int, default=8
        Minimum number of bins at any depth.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1,
                 initial_bins=256, bin_decay_rate=0.7, min_bins=8, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.initial_bins = initial_bins
        self.bin_decay_rate = bin_decay_rate
        self.min_bins = min_bins
        self.random_state = random_state
        
    def _get_bins_for_depth(self, depth):
        """Calculate number of bins for a given depth."""
        bins = int(self.initial_bins * (self.bin_decay_rate ** depth))
        return max(bins, self.min_bins)
    
    def _create_histogram(self, X, feature_idx, n_bins):
        """Create histogram bins for a feature."""
        feature_values = X[:, feature_idx]
        min_val, max_val = feature_values.min(), feature_values.max()
        
        if min_val == max_val:
            return None, None
        
        # Create bin edges
        bin_edges = np.linspace(min_val, max_val, n_bins + 1)
        # Digitize values into bins
        bin_indices = np.digitize(feature_values, bin_edges[1:-1])
        
        return bin_indices, bin_edges
    
    def _gini_impurity(self, y):
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def _find_best_split(self, X, y, depth):
        """Find the best split using histogram-based approach."""
        n_samples, n_features = X.shape
        n_bins = self._get_bins_for_depth(depth)
        
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        current_gini = self._gini_impurity(y)
        
        for feature_idx in range(n_features):
            bin_indices, bin_edges = self._create_histogram(X, feature_idx, n_bins)
            
            if bin_indices is None:
                continue
            
            # Try each bin boundary as a potential split
            for bin_idx in range(n_bins):
                left_mask = bin_indices <= bin_idx
                right_mask = ~left_mask
                
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                # Calculate weighted Gini impurity
                left_gini = self._gini_impurity(y[left_mask])
                right_gini = self._gini_impurity(y[right_mask])
                
                weighted_gini = (n_left * left_gini + n_right * right_gini) / n_samples
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = bin_edges[bin_idx + 1]
        
        return best_feature, best_threshold, best_gini
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            return {
                'is_leaf': True,
                'class': np.bincount(y).argmax(),
                'samples': n_samples
            }
        
        # Find best split
        best_feature, best_threshold, best_gini = self._find_best_split(X, y, depth)
        
        if best_feature is None:
            return {
                'is_leaf': True,
                'class': np.bincount(y).argmax(),
                'samples': n_samples
            }
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'is_leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree,
            'samples': n_samples
        }
    
    def _predict_sample(self, x, node):
        """Predict class for a single sample."""
        if node['is_leaf']:
            return node['class']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def fit(self, X_train, y_train):
        """
        Fit the adaptive histogram tree classifier.
        
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
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_train.shape[1]
        
        # Map classes to indices
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([self.class_to_idx_[c] for c in y_train])
        
        # Build the tree
        np.random.seed(self.random_state)
        self.tree_ = self._build_tree(X_train, y_encoded)
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fitted
        check_is_fitted(self, ['tree_', 'classes_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Predict for each sample
        predictions = np.array([self._predict_sample(x, self.tree_) for x in X_test])
        
        # Map indices back to original classes
        idx_to_class = {i: c for c, i in self.class_to_idx_.items()}
        return np.array([idx_to_class[idx] for idx in predictions])
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Class probabilities (one-hot encoded for this implementation).
        """
        check_is_fitted(self, ['tree_', 'classes_'])
        X_test = check_array(X_test)
        
        predictions = self.predict(X_test)
        proba = np.zeros((len(X_test), self.n_classes_))
        
        for i, pred in enumerate(predictions):
            class_idx = self.class_to_idx_[pred]
            proba[i, class_idx] = 1.0
        
        return proba