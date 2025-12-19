import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from collections import Counter


class MultiProjectionEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses multiple random feature subsets per split with ensemble
    voting to select the optimal split point across different feature space projections.
    
    Parameters
    ----------
    n_estimators : int, default=10
        The number of base decision trees in the ensemble.
    
    n_projections : int, default=5
        The number of random feature subsets to consider per split.
    
    max_features : float or int, default=0.5
        The number of features to consider for each projection.
        If float, then max_features is a fraction and int(max_features * n_features) 
        features are considered.
    
    max_depth : int, default=10
        The maximum depth of the decision trees.
    
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    
    random_state : int, default=None
        Controls the randomness of the estimator.
    """
    
    def __init__(self, n_estimators=10, n_projections=5, max_features=0.5,
                 max_depth=10, min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.n_projections = n_projections
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
    
    def _get_n_features(self, n_total_features):
        """Calculate the number of features to use per projection."""
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_total_features))
        else:
            return min(self.max_features, n_total_features)
    
    def _create_projection_tree(self, X, y, random_state):
        """
        Create a decision tree that uses multiple random projections per split.
        This is simulated by training a tree with a subset of features.
        """
        n_features = X.shape[1]
        n_features_subset = self._get_n_features(n_features)
        
        # Select random features for this projection
        rng = np.random.RandomState(random_state)
        feature_indices = rng.choice(n_features, size=n_features_subset, 
                                    replace=False)
        
        # Train tree on projected feature space
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=random_state,
            splitter='best'
        )
        
        X_projected = X[:, feature_indices]
        tree.fit(X_projected, y)
        
        return tree, feature_indices
    
    def _create_multi_projection_ensemble(self, X, y, base_seed):
        """
        Create an ensemble of trees, each using multiple random projections.
        """
        trees = []
        feature_sets = []
        
        rng = np.random.RandomState(base_seed)
        
        for _ in range(self.n_projections):
            seed = rng.randint(0, 10000)
            tree, features = self._create_projection_tree(X, y, seed)
            trees.append(tree)
            feature_sets.append(features)
        
        return trees, feature_sets
    
    def fit(self, X_train, y_train):
        """
        Fit the multi-projection ensemble classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        
        y_train : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        
        # Store classes
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_train.shape[1]
        
        # Initialize random state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Create ensemble of multi-projection trees
        self.estimators_ = []
        self.feature_sets_ = []
        
        rng = np.random.RandomState(self.random_state)
        
        for i in range(self.n_estimators):
            base_seed = rng.randint(0, 10000)
            trees, features = self._create_multi_projection_ensemble(
                X_train, y_train, base_seed
            )
            self.estimators_.append(trees)
            self.feature_sets_.append(features)
        
        return self
    
    def _predict_single_ensemble(self, X, trees, feature_sets):
        """
        Predict using a single multi-projection ensemble with voting.
        """
        predictions = []
        
        for tree, features in zip(trees, feature_sets):
            X_projected = X[:, features]
            pred = tree.predict(X_projected)
            predictions.append(pred)
        
        # Ensemble voting across projections
        predictions = np.array(predictions)
        voted_predictions = []
        
        for i in range(X.shape[0]):
            sample_predictions = predictions[:, i]
            # Majority vote
            counter = Counter(sample_predictions)
            voted_predictions.append(counter.most_common(1)[0][0])
        
        return np.array(voted_predictions)
    
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
        # Check if fit has been called
        check_is_fitted(self, ['estimators_', 'feature_sets_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(f"X_test has {X_test.shape[1]} features, "
                           f"but classifier was trained with {self.n_features_in_} features.")
        
        # Collect predictions from all ensembles
        all_predictions = []
        
        for trees, feature_sets in zip(self.estimators_, self.feature_sets_):
            pred = self._predict_single_ensemble(X_test, trees, feature_sets)
            all_predictions.append(pred)
        
        # Final ensemble voting across all multi-projection ensembles
        all_predictions = np.array(all_predictions)
        final_predictions = []
        
        for i in range(X_test.shape[0]):
            sample_predictions = all_predictions[:, i]
            counter = Counter(sample_predictions)
            final_predictions.append(counter.most_common(1)[0][0])
        
        return np.array(final_predictions)
    
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
            Class probabilities.
        """
        check_is_fitted(self, ['estimators_', 'feature_sets_'])
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(f"X_test has {X_test.shape[1]} features, "
                           f"but classifier was trained with {self.n_features_in_} features.")
        
        # Collect all predictions
        all_predictions = []
        
        for trees, feature_sets in zip(self.estimators_, self.feature_sets_):
            for tree, features in zip(trees, feature_sets):
                X_projected = X_test[:, features]
                pred = tree.predict(X_projected)
                all_predictions.append(pred)
        
        all_predictions = np.array(all_predictions)
        
        # Calculate probabilities based on voting
        n_samples = X_test.shape[0]
        probas = np.zeros((n_samples, self.n_classes_))
        
        for i in range(n_samples):
            sample_predictions = all_predictions[:, i]
            for pred in sample_predictions:
                class_idx = np.where(self.classes_ == pred)[0][0]
                probas[i, class_idx] += 1
        
        # Normalize
        probas /= probas.sum(axis=1, keepdims=True)
        
        return probas