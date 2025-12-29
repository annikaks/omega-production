import numpy as np
from sklearn.base import BaseEstimator


class GiniDecisionTreeClassifier(BaseEstimator):
    def __init__(self, max_depth=3):
        """
        Initialize the Decision Tree Classifier with Gini Impurity.
        
        Parameters:
        -----------
        max_depth : int, default=3
            Maximum depth of the tree
        """
        self.max_depth = max_depth
        self.tree_ = None
        self.n_classes_ = None
        self.n_features_ = None
    
    def _gini_impurity(self, y):
        """
        Calculate Gini impurity for a set of labels.
        
        Parameters:
        -----------
        y : array-like
            Labels
            
        Returns:
        --------
        float : Gini impurity value
        """
        if len(y) == 0:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini
    
    def _split_data(self, X, y, feature_idx, threshold):
        """
        Split data based on a feature and threshold.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features
        y : array-like of shape (n_samples,)
            Labels
        feature_idx : int
            Index of feature to split on
        threshold : float
            Threshold value for split
            
        Returns:
        --------
        tuple : (X_left, y_left, X_right, y_right)
        """
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        return (X[left_mask], y[left_mask], X[right_mask], y[right_mask])
    
    def _find_best_split(self, X, y):
        """
        Find the best split for the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features
        y : array-like of shape (n_samples,)
            Labels
            
        Returns:
        --------
        dict : Best split information (feature_idx, threshold, gini_gain)
        """
        best_gini_gain = -1
        best_feature_idx = None
        best_threshold = None
        current_gini = self._gini_impurity(y)
        n_samples = len(y)
        
        for feature_idx in range(X.shape[1]):
            # Get unique values for this feature
            thresholds = np.unique(X[:, feature_idx])
            
            # Try splits at midpoints between unique values
            for i in range(len(thresholds) - 1):
                threshold = (thresholds[i] + thresholds[i + 1]) / 2
                
                # Split the data
                X_left, y_left, X_right, y_right = self._split_data(
                    X, y, feature_idx, threshold
                )
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                # Calculate weighted Gini impurity
                n_left, n_right = len(y_left), len(y_right)
                gini_left = self._gini_impurity(y_left)
                gini_right = self._gini_impurity(y_right)
                weighted_gini = (n_left / n_samples) * gini_left + \
                               (n_right / n_samples) * gini_right
                
                # Calculate Gini gain
                gini_gain = current_gini - weighted_gini
                
                if gini_gain > best_gini_gain:
                    best_gini_gain = gini_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return {
            'feature_idx': best_feature_idx,
            'threshold': best_threshold,
            'gini_gain': best_gini_gain
        }
    
    def _build_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Features
        y : array-like of shape (n_samples,)
            Labels
        depth : int
            Current depth of the tree
            
        Returns:
        --------
        dict : Tree node
        """
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Calculate class distribution
        unique_classes, counts = np.unique(y, return_counts=True)
        predicted_class = unique_classes[np.argmax(counts)]
        
        # Create leaf node if stopping criteria met
        node = {
            'predicted_class': predicted_class,
            'n_samples': n_samples,
            'gini': self._gini_impurity(y)
        }
        
        # Check stopping criteria
        if depth >= self.max_depth or n_classes == 1 or n_samples < 2:
            return node
        
        # Find best split
        best_split = self._find_best_split(X, y)
        
        if best_split['feature_idx'] is None or best_split['gini_gain'] <= 0:
            return node
        
        # Split the data
        X_left, y_left, X_right, y_right = self._split_data(
            X, y, best_split['feature_idx'], best_split['threshold']
        )
        
        # Build subtrees
        node['feature_idx'] = best_split['feature_idx']
        node['threshold'] = best_split['threshold']
        node['left'] = self._build_tree(X_left, y_left, depth + 1)
        node['right'] = self._build_tree(X_right, y_right, depth + 1)
        
        return node
    
    def fit(self, X_train, y_train):
        """
        Fit the decision tree classifier.
        
        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            Training features
        y_train : array-like of shape (n_samples,)
            Training labels
            
        Returns:
        --------
        self : object
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.n_features_ = X_train.shape[1]
        self.n_classes_ = len(np.unique(y_train))
        
        # Build the tree
        self.tree_ = self._build_tree(X_train, y_train)
        
        return self
    
    def _predict_sample(self, x, node):
        """
        Predict class for a single sample.
        
        Parameters:
        -----------
        x : array-like of shape (n_features,)
            Single sample
        node : dict
            Current tree node
            
        Returns:
        --------
        int : Predicted class
        """
        # If leaf node, return prediction
        if 'feature_idx' not in node:
            return node['predicted_class']
        
        # Traverse tree
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X_test):
        """
        Predict classes for test samples.
        
        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            Test features
            
        Returns:
        --------
        array-like of shape (n_samples,) : Predicted classes
        """
        X_test = np.array(X_test)
        predictions = np.array([
            self._predict_sample(x, self.tree_) for x in X_test
        ])
        return predictions