import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class MultiResolutionTreeEnsemble(BaseEstimator, ClassifierMixin):
    """
    Multi-Resolution Tree Ensemble Classifier.
    
    Combines shallow trees (coarse patterns) with deep trees (fine-grained boundaries)
    to create a hierarchical ensemble that captures patterns at multiple resolutions.
    
    Parameters
    ----------
    n_shallow_trees : int, default=10
        Number of shallow trees for coarse pattern capture.
    
    n_deep_trees : int, default=10
        Number of deep trees for fine-grained refinement.
    
    shallow_max_depth : int, default=3
        Maximum depth for shallow trees.
    
    deep_max_depth : int, default=None
        Maximum depth for deep trees (None means unlimited).
    
    shallow_weight : float, default=0.4
        Weight for shallow tree predictions (deep_weight = 1 - shallow_weight).
    
    max_features : str or float, default='sqrt'
        Number of features to consider for splits.
    
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node.
    
    random_state : int, default=None
        Random seed for reproducibility.
    
    bootstrap : bool, default=True
        Whether to use bootstrap sampling for trees.
    """
    
    def __init__(
        self,
        n_shallow_trees=10,
        n_deep_trees=10,
        shallow_max_depth=3,
        deep_max_depth=None,
        shallow_weight=0.4,
        max_features='sqrt',
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=None,
        bootstrap=True
    ):
        self.n_shallow_trees = n_shallow_trees
        self.n_deep_trees = n_deep_trees
        self.shallow_max_depth = shallow_max_depth
        self.deep_max_depth = deep_max_depth
        self.shallow_weight = shallow_weight
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.bootstrap = bootstrap
    
    def _create_tree(self, max_depth, random_state):
        """Create a decision tree with specified parameters."""
        return DecisionTreeClassifier(
            max_depth=max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=random_state
        )
    
    def _bootstrap_sample(self, X, y, random_state):
        """Generate bootstrap sample of data."""
        rng = np.random.RandomState(random_state)
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X_train, y_train):
        """
        Fit the multi-resolution tree ensemble.
        
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
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Initialize shallow trees (coarse patterns)
        self.shallow_trees_ = []
        for i in range(self.n_shallow_trees):
            tree_random_state = rng.randint(0, 2**31 - 1)
            tree = self._create_tree(self.shallow_max_depth, tree_random_state)
            
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(
                    X_train, y_train, tree_random_state
                )
            else:
                X_sample, y_sample = X_train, y_train
            
            tree.fit(X_sample, y_sample)
            self.shallow_trees_.append(tree)
        
        # Initialize deep trees (fine-grained patterns)
        self.deep_trees_ = []
        for i in range(self.n_deep_trees):
            tree_random_state = rng.randint(0, 2**31 - 1)
            tree = self._create_tree(self.deep_max_depth, tree_random_state)
            
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(
                    X_train, y_train, tree_random_state
                )
            else:
                X_sample, y_sample = X_train, y_train
            
            tree.fit(X_sample, y_sample)
            self.deep_trees_.append(tree)
        
        return self
    
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
        
        # Aggregate predictions from shallow trees
        shallow_proba = np.zeros((n_samples, self.n_classes_))
        for tree in self.shallow_trees_:
            shallow_proba += tree.predict_proba(X_test)
        shallow_proba /= self.n_shallow_trees
        
        # Aggregate predictions from deep trees
        deep_proba = np.zeros((n_samples, self.n_classes_))
        for tree in self.deep_trees_:
            deep_proba += tree.predict_proba(X_test)
        deep_proba /= self.n_deep_trees
        
        # Combine predictions with weighted average
        combined_proba = (
            self.shallow_weight * shallow_proba +
            (1 - self.shallow_weight) * deep_proba
        )
        
        return combined_proba
    
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
        Get feature importances from all trees.
        
        Returns
        -------
        importance : dict
            Dictionary with 'shallow', 'deep', and 'combined' feature importances.
        """
        check_is_fitted(self)
        
        # Shallow tree importances
        shallow_importance = np.zeros(self.n_features_in_)
        for tree in self.shallow_trees_:
            shallow_importance += tree.feature_importances_
        shallow_importance /= self.n_shallow_trees
        
        # Deep tree importances
        deep_importance = np.zeros(self.n_features_in_)
        for tree in self.deep_trees_:
            deep_importance += tree.feature_importances_
        deep_importance /= self.n_deep_trees
        
        # Combined importance
        combined_importance = (
            self.shallow_weight * shallow_importance +
            (1 - self.shallow_weight) * deep_importance
        )
        
        return {
            'shallow': shallow_importance,
            'deep': deep_importance,
            'combined': combined_importance
        }