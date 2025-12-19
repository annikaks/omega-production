import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist


class KDEWeightedOOBForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Random Forest classifier that uses kernel density estimation on out-of-bag
    samples to weight tree votes based on local prediction confidence regions.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    
    max_depth : int, default=None
        The maximum depth of the trees.
    
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    
    max_features : str or int, default='sqrt'
        The number of features to consider when looking for the best split.
    
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees.
    
    kde_bandwidth : float, default=1.0
        The bandwidth of the kernel density estimator.
    
    random_state : int, default=None
        Controls the randomness of the estimator.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features='sqrt', bootstrap=True, kde_bandwidth=1.0,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.kde_bandwidth = kde_bandwidth
        self.random_state = random_state
    
    def _bootstrap_sample(self, X, y, rng):
        """Generate bootstrap sample and return indices."""
        n_samples = X.shape[0]
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        oob_indices = np.setdiff1d(np.arange(n_samples), np.unique(indices))
        return indices, oob_indices
    
    def _fit_kde_for_tree(self, X_oob, y_oob, tree_pred_oob, class_label):
        """
        Fit KDE on OOB samples where the tree correctly predicted the class.
        
        Parameters
        ----------
        X_oob : array-like
            Out-of-bag samples
        y_oob : array-like
            True labels for OOB samples
        tree_pred_oob : array-like
            Tree predictions for OOB samples
        class_label : int
            The class label to fit KDE for
        
        Returns
        -------
        kde : KernelDensity or None
            Fitted KDE model or None if no correct predictions
        """
        # Find samples where tree correctly predicted this class
        correct_mask = (tree_pred_oob == class_label) & (y_oob == class_label)
        
        if np.sum(correct_mask) < 2:
            return None
        
        X_correct = X_oob[correct_mask]
        
        # Fit KDE on correctly predicted samples
        kde = KernelDensity(bandwidth=self.kde_bandwidth, kernel='gaussian')
        kde.fit(X_correct)
        
        return kde
    
    def fit(self, X_train, y_train):
        """
        Build a forest of trees with KDE-based weighting from the training set.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The training input samples.
        
        y_train : array-like of shape (n_samples,)
            The target values (class labels).
        
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
        
        # Initialize storage
        self.trees_ = []
        self.kdes_ = []  # List of dicts mapping class -> KDE for each tree
        
        # Build trees
        for i in range(self.n_estimators):
            # Create bootstrap sample
            if self.bootstrap:
                boot_indices, oob_indices = self._bootstrap_sample(X_train, y_train, rng)
                X_boot = X_train[boot_indices]
                y_boot = y_train[boot_indices]
            else:
                X_boot = X_train
                y_boot = y_train
                oob_indices = np.array([])
            
            # Train tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=rng.randint(0, 10000)
            )
            tree.fit(X_boot, y_boot)
            self.trees_.append(tree)
            
            # Fit KDEs on OOB samples for each class
            tree_kdes = {}
            if len(oob_indices) > 0:
                X_oob = X_train[oob_indices]
                y_oob = y_train[oob_indices]
                tree_pred_oob = tree.predict(X_oob)
                
                for class_label in self.classes_:
                    kde = self._fit_kde_for_tree(X_oob, y_oob, tree_pred_oob, class_label)
                    tree_kdes[class_label] = kde
            
            self.kdes_.append(tree_kdes)
        
        return self
    
    def _compute_tree_weights(self, X_test):
        """
        Compute weights for each tree based on KDE confidence.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        weights : array of shape (n_samples, n_estimators, n_classes)
            Weights for each tree and class for each test sample.
        """
        n_samples = X_test.shape[0]
        weights = np.ones((n_samples, self.n_estimators, self.n_classes_))
        
        for tree_idx, (tree, tree_kdes) in enumerate(zip(self.trees_, self.kdes_)):
            # Get tree predictions
            tree_preds = tree.predict(X_test)
            
            # Compute weights based on KDE scores
            for class_idx, class_label in enumerate(self.classes_):
                kde = tree_kdes.get(class_label)
                
                if kde is not None:
                    # Compute log density for samples where tree predicts this class
                    mask = tree_preds == class_label
                    if np.any(mask):
                        log_density = kde.score_samples(X_test[mask])
                        # Convert log density to weight (exponential to get density)
                        # Add small constant to avoid zero weights
                        density_weights = np.exp(log_density)
                        weights[mask, tree_idx, class_idx] = density_weights + 1e-10
                    else:
                        weights[:, tree_idx, class_idx] = 1e-10
                else:
                    # No KDE available, use uniform weight
                    weights[:, tree_idx, class_idx] = 1.0
        
        return weights
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X_test) for tree in self.trees_])
        
        # Compute weights based on KDE confidence
        weights = self._compute_tree_weights(X_test)
        
        # Weighted voting
        weighted_votes = np.zeros((n_samples, self.n_classes_))
        
        for sample_idx in range(n_samples):
            for tree_idx in range(self.n_estimators):
                pred_class = tree_predictions[tree_idx, sample_idx]
                class_idx = np.where(self.classes_ == pred_class)[0][0]
                
                # Add weighted vote
                weight = weights[sample_idx, tree_idx, class_idx]
                weighted_votes[sample_idx, class_idx] += weight
        
        # Return class with highest weighted vote
        y_pred = self.classes_[np.argmax(weighted_votes, axis=1)]
        
        return y_pred
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X_test) for tree in self.trees_])
        
        # Compute weights based on KDE confidence
        weights = self._compute_tree_weights(X_test)
        
        # Weighted voting
        weighted_votes = np.zeros((n_samples, self.n_classes_))
        
        for sample_idx in range(n_samples):
            for tree_idx in range(self.n_estimators):
                pred_class = tree_predictions[tree_idx, sample_idx]
                class_idx = np.where(self.classes_ == pred_class)[0][0]
                
                # Add weighted vote
                weight = weights[sample_idx, tree_idx, class_idx]
                weighted_votes[sample_idx, class_idx] += weight
        
        # Normalize to get probabilities
        proba = weighted_votes / weighted_votes.sum(axis=1, keepdims=True)
        
        return proba