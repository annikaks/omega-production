import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class HierarchicalDepthEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Hierarchical tree depth allocation classifier where shallow trees capture 
    coarse patterns and deep trees refine fine-grained decision boundaries.
    
    Parameters
    ----------
    depth_levels : list of int, default=[1, 3, 5, 10, 20]
        List of maximum depths for trees in the hierarchy, ordered from shallow to deep.
    
    n_trees_per_level : int or list of int, default=3
        Number of trees at each depth level. If int, same number for all levels.
        If list, must match length of depth_levels.
    
    weight_strategy : str, default='adaptive'
        Strategy for weighting predictions from different levels:
        - 'uniform': Equal weights for all levels
        - 'adaptive': Weights based on confidence/entropy
        - 'depth_weighted': Higher weights for deeper trees
    
    bootstrap : bool, default=True
        Whether to use bootstrap sampling for each tree.
    
    random_state : int, default=None
        Random state for reproducibility.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    
    trees_ : list of list of DecisionTreeClassifier
        Hierarchical structure of trees organized by depth level.
    
    level_weights_ : ndarray
        Weights for each depth level.
    """
    
    def __init__(self, depth_levels=None, n_trees_per_level=3, 
                 weight_strategy='adaptive', bootstrap=True, random_state=None):
        self.depth_levels = depth_levels if depth_levels is not None else [1, 3, 5, 10, 20]
        self.n_trees_per_level = n_trees_per_level
        self.weight_strategy = weight_strategy
        self.bootstrap = bootstrap
        self.random_state = random_state
    
    def fit(self, X_train, y_train):
        """
        Fit the hierarchical ensemble of decision trees.
        
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
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_train.shape[1]
        
        # Setup number of trees per level
        if isinstance(self.n_trees_per_level, int):
            n_trees = [self.n_trees_per_level] * len(self.depth_levels)
        else:
            n_trees = self.n_trees_per_level
            if len(n_trees) != len(self.depth_levels):
                raise ValueError("Length of n_trees_per_level must match depth_levels")
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Build hierarchical tree structure
        self.trees_ = []
        self.level_oob_scores_ = []
        
        for level_idx, (depth, n_tree) in enumerate(zip(self.depth_levels, n_trees)):
            level_trees = []
            level_predictions = []
            
            for tree_idx in range(n_tree):
                # Create tree with specified depth
                tree = DecisionTreeClassifier(
                    max_depth=depth,
                    random_state=rng.randint(0, 10000) if self.random_state is not None else None
                )
                
                # Bootstrap sampling if enabled
                if self.bootstrap:
                    n_samples = X_train.shape[0]
                    indices = rng.choice(n_samples, size=n_samples, replace=True)
                    X_boot = X_train[indices]
                    y_boot = y_train[indices]
                    
                    # Fit tree
                    tree.fit(X_boot, y_boot)
                    
                    # Calculate OOB score for adaptive weighting
                    oob_mask = np.ones(n_samples, dtype=bool)
                    oob_mask[indices] = False
                    if np.sum(oob_mask) > 0:
                        oob_pred = tree.predict(X_train[oob_mask])
                        oob_acc = np.mean(oob_pred == y_train[oob_mask])
                        level_predictions.append(oob_acc)
                else:
                    tree.fit(X_train, y_train)
                    # Use training accuracy as proxy
                    train_pred = tree.predict(X_train)
                    train_acc = np.mean(train_pred == y_train)
                    level_predictions.append(train_acc)
                
                level_trees.append(tree)
            
            self.trees_.append(level_trees)
            # Store average performance for this level
            self.level_oob_scores_.append(np.mean(level_predictions) if level_predictions else 0.5)
        
        # Calculate level weights based on strategy
        self._calculate_level_weights()
        
        return self
    
    def _calculate_level_weights(self):
        """Calculate weights for each depth level based on the chosen strategy."""
        n_levels = len(self.depth_levels)
        
        if self.weight_strategy == 'uniform':
            self.level_weights_ = np.ones(n_levels) / n_levels
        
        elif self.weight_strategy == 'depth_weighted':
            # Higher weights for deeper trees
            weights = np.array(self.depth_levels, dtype=float)
            self.level_weights_ = weights / np.sum(weights)
        
        elif self.weight_strategy == 'adaptive':
            # Weights based on OOB performance
            weights = np.array(self.level_oob_scores_)
            # Ensure no zero weights
            weights = np.maximum(weights, 0.1)
            self.level_weights_ = weights / np.sum(weights)
        
        else:
            raise ValueError(f"Unknown weight_strategy: {self.weight_strategy}")
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        aggregated_proba = np.zeros((n_samples, self.n_classes_))
        
        # Aggregate predictions from all levels
        for level_idx, level_trees in enumerate(self.trees_):
            level_proba = np.zeros((n_samples, self.n_classes_))
            
            # Average predictions from trees at this level
            for tree in level_trees:
                tree_proba = tree.predict_proba(X_test)
                level_proba += tree_proba
            
            level_proba /= len(level_trees)
            
            # Weight by level importance
            aggregated_proba += self.level_weights_[level_idx] * level_proba
        
        return aggregated_proba
    
    def predict(self, X_test):
        """
        Predict class labels for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def get_feature_importance(self):
        """
        Calculate aggregated feature importance across all trees.
        
        Returns
        -------
        importance : ndarray of shape (n_features,)
            Feature importances.
        """
        check_is_fitted(self)
        
        importance = np.zeros(self.n_features_in_)
        total_weight = 0
        
        for level_idx, level_trees in enumerate(self.trees_):
            level_weight = self.level_weights_[level_idx]
            for tree in level_trees:
                importance += level_weight * tree.feature_importances_
                total_weight += level_weight
        
        return importance / total_weight if total_weight > 0 else importance