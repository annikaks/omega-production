import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class NoisyThresholdEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that injects controlled noise into decision tree split thresholds
    during training and averages predictions across multiple noisy versions for
    robust bagging-like behavior.
    
    Parameters
    ----------
    n_estimators : int, default=10
        The number of noisy tree versions to train.
    
    noise_scale : float, default=0.1
        The scale of Gaussian noise to inject into split thresholds.
        Noise is proportional to the feature range.
    
    max_depth : int, default=None
        The maximum depth of the decision trees.
    
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_estimators=10, noise_scale=0.1, max_depth=None,
                 min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.noise_scale = noise_scale
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
    
    def _inject_threshold_noise(self, tree, feature_ranges, rng):
        """
        Inject noise into the split thresholds of a trained decision tree.
        
        Parameters
        ----------
        tree : DecisionTreeClassifier
            The trained decision tree.
        feature_ranges : array-like
            The range (max - min) for each feature.
        rng : np.random.Generator
            Random number generator.
        """
        tree_structure = tree.tree_
        n_nodes = tree_structure.node_count
        
        # Iterate through all nodes and add noise to split thresholds
        for node_id in range(n_nodes):
            # Only modify internal nodes (not leaves)
            if tree_structure.feature[node_id] != -2:  # -2 indicates a leaf
                feature_idx = tree_structure.feature[node_id]
                current_threshold = tree_structure.threshold[node_id]
                
                # Add Gaussian noise proportional to feature range
                noise = rng.normal(0, self.noise_scale * feature_ranges[feature_idx])
                tree_structure.threshold[node_id] = current_threshold + noise
    
    def fit(self, X_train, y_train):
        """
        Fit the noisy threshold ensemble classifier.
        
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
        
        # Calculate feature ranges for noise scaling
        self.feature_ranges_ = np.ptp(X_train, axis=0)
        # Avoid division by zero for constant features
        self.feature_ranges_[self.feature_ranges_ == 0] = 1.0
        
        # Initialize random number generator
        rng = np.random.default_rng(self.random_state)
        
        # Train multiple trees with noisy thresholds
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # Train a base decision tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=rng.integers(0, 2**31) if self.random_state is not None else None
            )
            tree.fit(X_train, y_train)
            
            # Inject noise into thresholds (except for the first tree as baseline)
            if i > 0:
                self._inject_threshold_noise(tree, self.feature_ranges_, rng)
            
            self.estimators_.append(tree)
        
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
            The class probabilities of the input samples.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        # Collect predictions from all noisy estimators
        all_probas = np.zeros((X_test.shape[0], self.n_classes_))
        
        for estimator in self.estimators_:
            all_probas += estimator.predict_proba(X_test)
        
        # Average predictions across all estimators
        avg_probas = all_probas / self.n_estimators
        
        return avg_probas
    
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
        # Get averaged probabilities
        probas = self.predict_proba(X_test)
        
        # Return class with highest probability
        return self.classes_[np.argmax(probas, axis=1)]