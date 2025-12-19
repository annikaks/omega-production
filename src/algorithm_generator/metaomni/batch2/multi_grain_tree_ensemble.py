import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from typing import List, Tuple, Optional


class MultiGrainTreeEnsemble(BaseEstimator, ClassifierMixin):
    """
    Multi-grain tree ensemble that combines shallow trees for coarse patterns
    with deep trees for fine-grained interactions.
    
    Parameters
    ----------
    n_shallow_trees : int, default=50
        Number of shallow trees for coarse patterns.
    
    n_deep_trees : int, default=50
        Number of deep trees for fine-grained interactions.
    
    shallow_max_depth : int, default=3
        Maximum depth for shallow trees.
    
    deep_max_depth : int, default=15
        Maximum depth for deep trees.
    
    max_features : str or float, default='sqrt'
        Number of features to consider for splits.
    
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    
    min_samples_leaf : int, default=1
        Minimum samples required at a leaf node.
    
    bootstrap : bool, default=True
        Whether to use bootstrap samples.
    
    random_state : int, optional
        Random state for reproducibility.
    
    voting : str, default='soft'
        Voting strategy: 'soft' for probability averaging or 'hard' for majority vote.
    """
    
    def __init__(
        self,
        n_shallow_trees: int = 50,
        n_deep_trees: int = 50,
        shallow_max_depth: int = 3,
        deep_max_depth: int = 15,
        max_features: str = 'sqrt',
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
        voting: str = 'soft'
    ):
        self.n_shallow_trees = n_shallow_trees
        self.n_deep_trees = n_deep_trees
        self.shallow_max_depth = shallow_max_depth
        self.deep_max_depth = deep_max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.voting = voting
        
    def _create_tree(self, max_depth: int, random_state: int) -> DecisionTreeClassifier:
        """Create a decision tree with specified parameters."""
        return DecisionTreeClassifier(
            max_depth=max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=random_state
        )
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray, 
                         random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate bootstrap sample from training data."""
        rng = np.random.RandomState(random_state)
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the multi-grain tree ensemble.
        
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
        
        # Initialize tree lists
        self.shallow_trees_: List[DecisionTreeClassifier] = []
        self.deep_trees_: List[DecisionTreeClassifier] = []
        
        # Train shallow trees (coarse patterns)
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
        
        # Train deep trees (fine-grained interactions)
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
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        n_trees = self.n_shallow_trees + self.n_deep_trees
        
        # Accumulate predictions from all trees
        proba_sum = np.zeros((n_samples, self.n_classes_))
        
        # Add predictions from shallow trees
        for tree in self.shallow_trees_:
            proba_sum += tree.predict_proba(X_test)
        
        # Add predictions from deep trees
        for tree in self.deep_trees_:
            proba_sum += tree.predict_proba(X_test)
        
        # Average probabilities
        proba = proba_sum / n_trees
        
        return proba
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        if self.voting == 'soft':
            # Use probability-based prediction
            proba = self.predict_proba(X_test)
            return self.classes_[np.argmax(proba, axis=1)]
        else:
            # Hard voting: majority vote
            n_samples = X_test.shape[0]
            predictions = np.zeros((n_samples, self.n_shallow_trees + self.n_deep_trees))
            
            # Collect predictions from shallow trees
            for i, tree in enumerate(self.shallow_trees_):
                predictions[:, i] = tree.predict(X_test)
            
            # Collect predictions from deep trees
            offset = self.n_shallow_trees
            for i, tree in enumerate(self.deep_trees_):
                predictions[:, offset + i] = tree.predict(X_test)
            
            # Majority vote
            y_pred = np.zeros(n_samples, dtype=self.classes_.dtype)
            for i in range(n_samples):
                y_pred[i] = np.bincount(predictions[i].astype(int)).argmax()
            
            return y_pred
    
    def get_feature_importances(self) -> np.ndarray:
        """
        Compute average feature importances across all trees.
        
        Returns
        -------
        importances : ndarray of shape (n_features,)
            Feature importances.
        """
        check_is_fitted(self)
        
        importances = np.zeros(self.n_features_in_)
        n_trees = self.n_shallow_trees + self.n_deep_trees
        
        for tree in self.shallow_trees_:
            importances += tree.feature_importances_
        
        for tree in self.deep_trees_:
            importances += tree.feature_importances_
        
        return importances / n_trees