import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import entropy


class AdaptiveSubspaceForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Random Forest with adaptive feature subspace projection based on local data complexity.
    
    Each tree samples a different number of features based on the complexity of the
    data subset it receives, measured by entropy and feature variance.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    
    min_features : int, default=1
        Minimum number of features to sample per tree.
    
    max_features : int or None, default=None
        Maximum number of features to sample per tree.
        If None, uses all features.
    
    complexity_metric : str, default='entropy'
        Metric to measure data complexity: 'entropy', 'variance', or 'combined'.
    
    max_depth : int or None, default=None
        Maximum depth of each tree.
    
    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
    
    bootstrap : bool, default=True
        Whether to use bootstrap samples when building trees.
    
    random_state : int or None, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_estimators=100, min_features=1, max_features=None,
                 complexity_metric='entropy', max_depth=None, min_samples_split=2,
                 bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.min_features = min_features
        self.max_features = max_features
        self.complexity_metric = complexity_metric
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.random_state = random_state
    
    def _compute_data_complexity(self, X, y):
        """
        Compute the complexity of the data to determine feature subspace size.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        complexity : float
            Normalized complexity score between 0 and 1.
        """
        if self.complexity_metric == 'entropy':
            # Compute label entropy
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            label_entropy = entropy(probs)
            # Normalize by max possible entropy
            max_entropy = np.log(len(np.unique(y)))
            complexity = label_entropy / max_entropy if max_entropy > 0 else 0.5
            
        elif self.complexity_metric == 'variance':
            # Compute average normalized variance across features
            variances = np.var(X, axis=0)
            # Normalize variances
            mean_variance = np.mean(variances)
            max_variance = np.max(variances) if np.max(variances) > 0 else 1.0
            complexity = mean_variance / max_variance
            
        elif self.complexity_metric == 'combined':
            # Combine entropy and variance
            _, counts = np.unique(y, return_counts=True)
            probs = counts / len(y)
            label_entropy = entropy(probs)
            max_entropy = np.log(len(np.unique(y)))
            entropy_score = label_entropy / max_entropy if max_entropy > 0 else 0.5
            
            variances = np.var(X, axis=0)
            mean_variance = np.mean(variances)
            max_variance = np.max(variances) if np.max(variances) > 0 else 1.0
            variance_score = mean_variance / max_variance
            
            complexity = 0.5 * entropy_score + 0.5 * variance_score
        else:
            complexity = 0.5
        
        return np.clip(complexity, 0.0, 1.0)
    
    def _determine_n_features(self, complexity, n_total_features):
        """
        Determine the number of features to sample based on complexity.
        
        Higher complexity -> more features needed.
        
        Parameters
        ----------
        complexity : float
            Complexity score between 0 and 1.
        n_total_features : int
            Total number of features available.
        
        Returns
        -------
        n_features : int
            Number of features to sample.
        """
        max_feat = self.max_features if self.max_features is not None else n_total_features
        max_feat = min(max_feat, n_total_features)
        
        # Map complexity to feature count
        # Higher complexity -> sample more features
        feature_range = max_feat - self.min_features
        n_features = int(self.min_features + complexity * feature_range)
        
        return max(self.min_features, min(n_features, max_feat))
    
    def fit(self, X_train, y_train):
        """
        Build a forest of trees with adaptive feature subspace projection.
        
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
        
        # Store trees and their feature indices
        self.trees_ = []
        self.feature_indices_ = []
        self.n_features_per_tree_ = []
        
        n_samples = X_train.shape[0]
        
        for i in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                X_subset = X_train[indices]
                y_subset = y_train[indices]
            else:
                X_subset = X_train
                y_subset = y_train
            
            # Compute complexity of this subset
            complexity = self._compute_data_complexity(X_subset, y_subset)
            
            # Determine number of features based on complexity
            n_features = self._determine_n_features(complexity, self.n_features_in_)
            self.n_features_per_tree_.append(n_features)
            
            # Randomly select features
            feature_idx = rng.choice(self.n_features_in_, size=n_features, replace=False)
            feature_idx = np.sort(feature_idx)
            self.feature_indices_.append(feature_idx)
            
            # Train tree on selected features
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=rng.randint(0, 10000)
            )
            tree.fit(X_subset[:, feature_idx], y_subset)
            self.trees_.append(tree)
        
        return self
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(f"X_test has {X_test.shape[1]} features, "
                           f"but model was trained with {self.n_features_in_} features.")
        
        # Aggregate predictions from all trees
        all_proba = np.zeros((X_test.shape[0], self.n_classes_))
        
        for tree, feature_idx in zip(self.trees_, self.feature_indices_):
            X_subset = X_test[:, feature_idx]
            proba = tree.predict_proba(X_subset)
            all_proba += proba
        
        # Average probabilities
        all_proba /= self.n_estimators
        
        return all_proba
    
    def predict(self, X_test):
        """
        Predict class labels for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def get_feature_importance(self):
        """
        Compute feature importance based on usage frequency and tree importances.
        
        Returns
        -------
        importances : array of shape (n_features,)
            Feature importances.
        """
        check_is_fitted(self)
        
        importances = np.zeros(self.n_features_in_)
        
        for tree, feature_idx in zip(self.trees_, self.feature_indices_):
            tree_importances = tree.feature_importances_
            importances[feature_idx] += tree_importances
        
        # Normalize
        if importances.sum() > 0:
            importances /= importances.sum()
        
        return importances