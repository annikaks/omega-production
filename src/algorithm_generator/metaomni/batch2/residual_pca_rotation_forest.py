import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler


class ResidualPCARotationForest(BaseEstimator, ClassifierMixin):
    """
    A classifier that dynamically rotates feature subspaces between trees using PCA
    on residuals from previous tree subsets.
    
    Parameters
    ----------
    n_estimators : int, default=10
        The number of trees in the forest.
    
    n_features_per_subset : int or float, default=3
        Number of features per subset. If float, represents the proportion of features.
    
    max_depth : int, default=None
        Maximum depth of the decision trees.
    
    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
    
    subset_size : int, default=5
        Number of trees to use for computing residuals before rotation.
    
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_estimators=10, n_features_per_subset=3, max_depth=None,
                 min_samples_split=2, subset_size=5, random_state=None):
        self.n_estimators = n_estimators
        self.n_features_per_subset = n_features_per_subset
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subset_size = subset_size
        self.random_state = random_state
    
    def _compute_residuals(self, X, y, trees, rotations, feature_subsets):
        """Compute residuals from a subset of trees."""
        predictions = np.zeros((X.shape[0], len(self.classes_)))
        
        for tree, rotation, features in zip(trees, rotations, feature_subsets):
            X_subset = X[:, features]
            X_rotated = X_subset @ rotation
            
            # Get probability predictions
            proba = tree.predict_proba(X_rotated)
            predictions += proba
        
        # Average predictions
        predictions /= len(trees)
        
        # Compute residuals (difference from true labels)
        y_one_hot = np.zeros((len(y), len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            y_one_hot[y == cls, i] = 1
        
        residuals = y_one_hot - predictions
        return residuals
    
    def _create_rotation_matrix(self, X_subset, residuals=None):
        """Create rotation matrix using PCA, optionally weighted by residuals."""
        n_samples, n_features = X_subset.shape
        
        if residuals is not None and residuals.size > 0:
            # Weight samples by residual magnitude
            weights = np.abs(residuals).sum(axis=1)
            weights = weights / (weights.sum() + 1e-10)
            
            # Center data with weights
            mean = np.average(X_subset, axis=0, weights=weights)
            X_centered = X_subset - mean
            
            # Weighted covariance
            weighted_X = X_centered * np.sqrt(weights)[:, np.newaxis]
            
            # Apply PCA
            n_components = min(n_features, n_samples)
            pca = PCA(n_components=n_components, random_state=self.random_state)
            pca.fit(weighted_X)
            rotation = pca.components_.T
        else:
            # Standard PCA without weighting
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            
            n_components = min(n_features, n_samples)
            pca = PCA(n_components=n_components, random_state=self.random_state)
            pca.fit(X_scaled)
            rotation = pca.components_.T
        
        return rotation
    
    def fit(self, X_train, y_train):
        """
        Fit the Residual PCA Rotation Forest classifier.
        
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
        self.n_features_in_ = X_train.shape[1]
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Determine features per subset
        if isinstance(self.n_features_per_subset, float):
            n_features_subset = max(1, int(self.n_features_per_subset * self.n_features_in_))
        else:
            n_features_subset = min(self.n_features_per_subset, self.n_features_in_)
        
        # Initialize storage
        self.trees_ = []
        self.rotations_ = []
        self.feature_subsets_ = []
        
        # Track trees for residual computation
        subset_trees = []
        subset_rotations = []
        subset_features = []
        
        for i in range(self.n_estimators):
            # Select random feature subset
            features = rng.choice(self.n_features_in_, size=n_features_subset, replace=False)
            features = np.sort(features)
            
            X_subset = X_train[:, features]
            
            # Compute residuals from previous subset of trees
            if len(subset_trees) >= self.subset_size:
                residuals = self._compute_residuals(
                    X_train, y_train, subset_trees, subset_rotations, subset_features
                )
                # Reset subset
                subset_trees = []
                subset_rotations = []
                subset_features = []
            elif len(subset_trees) > 0:
                residuals = self._compute_residuals(
                    X_train, y_train, subset_trees, subset_rotations, subset_features
                )
            else:
                residuals = None
            
            # Create rotation matrix based on residuals
            rotation = self._create_rotation_matrix(X_subset, residuals)
            
            # Apply rotation
            X_rotated = X_subset @ rotation
            
            # Train decision tree on rotated features
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=rng.randint(0, 10000)
            )
            tree.fit(X_rotated, y_train)
            
            # Store tree, rotation, and features
            self.trees_.append(tree)
            self.rotations_.append(rotation)
            self.feature_subsets_.append(features)
            
            # Add to current subset
            subset_trees.append(tree)
            subset_rotations.append(rotation)
            subset_features.append(features)
        
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
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_test.shape[1]}")
        
        # Aggregate predictions from all trees
        predictions = np.zeros((X_test.shape[0], len(self.classes_)))
        
        for tree, rotation, features in zip(self.trees_, self.rotations_, self.feature_subsets_):
            X_subset = X_test[:, features]
            X_rotated = X_subset @ rotation
            predictions += tree.predict_proba(X_rotated)
        
        # Average predictions
        predictions /= self.n_estimators
        
        return predictions
    
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