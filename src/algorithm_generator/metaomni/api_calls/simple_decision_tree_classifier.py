from sklearn.base import BaseEstimator
import numpy as np


class SimpleDecisionTreeClassifier(BaseEstimator):
    def __init__(self, max_depth=5, min_samples_split=2):
        """
        Simple Decision Tree Classifier
        
        Parameters:
        -----------
        max_depth : int, default=5
            Maximum depth of the tree
        min_samples_split : int, default=2
            Minimum number of samples required to split a node
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None
        
    def _gini_impurity(self, y):
        """Calculate Gini impurity for a set of labels"""
        if len(y) == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _split_data(self, X, y, feature_idx, threshold):
        """Split data based on a feature and threshold"""
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        return (X[left_mask], y[left_mask], X[right_mask], y[right_mask])
    
    def _find_best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        n_samples, n_features = X.shape
        
        current_gini = self._gini_impurity(y)
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split_data(
                    X, y, feature_idx, threshold
                )
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                # Calculate weighted Gini impurity
                n_left, n_right = len(y_left), len(y_right)
                weighted_gini = (n_left / n_samples) * self._gini_impurity(y_left) + \
                               (n_right / n_samples) * self._gini_impurity(y_right)
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            # Create leaf node
            leaf_value = np.bincount(y.astype(int)).argmax()
            return {'type': 'leaf', 'value': leaf_value}
        
        # Find best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            # No valid split found, create leaf
            leaf_value = np.bincount(y.astype(int)).argmax()
            return {'type': 'leaf', 'value': leaf_value}
        
        # Split data
        X_left, y_left, X_right, y_right = self._split_data(
            X, y, best_feature, best_threshold
        )
        
        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        
        return {
            'type': 'node',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X_train, y_train):
        """
        Build the decision tree from training data
        
        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.tree_ = self._build_tree(X_train, y_train)
        self.classes_ = np.unique(y_train)
        
        return self
    
    def _predict_sample(self, x, node):
        """Predict class for a single sample"""
        if node['type'] == 'leaf':
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test
        
        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        X_test = np.array(X_test)
        predictions = np.array([
            self._predict_sample(x, self.tree_) for x in X_test
        ])
        return predictions