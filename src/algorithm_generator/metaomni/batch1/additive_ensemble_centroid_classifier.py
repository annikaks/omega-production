import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import pairwise_distances


class AdditiveEnsembleCentroidClassifier(BaseEstimator, ClassifierMixin):
    """
    Additive Ensemble of Centroid Classifiers on Complementary Feature Subspaces.
    
    This classifier trains multiple centroid-based classifiers on different feature
    subspaces and combines their distance predictions using a linear combination.
    
    Parameters
    ----------
    n_estimators : int, default=10
        Number of centroid classifiers in the ensemble.
    
    feature_subset_size : float or int, default=0.5
        If float, represents the proportion of features to use per estimator.
        If int, represents the absolute number of features to use.
    
    metric : str, default='euclidean'
        Distance metric to use for computing distances to centroids.
        
    random_state : int, default=None
        Random state for reproducibility of feature subspace selection.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    
    centroids_ : list of dict
        List of dictionaries mapping class labels to centroids for each estimator.
    
    feature_subspaces_ : list of ndarray
        List of feature indices used by each estimator.
    """
    
    def __init__(self, n_estimators=10, feature_subset_size=0.5, 
                 metric='euclidean', random_state=None):
        self.n_estimators = n_estimators
        self.feature_subset_size = feature_subset_size
        self.metric = metric
        self.random_state = random_state
    
    def _generate_feature_subspaces(self, n_features):
        """Generate complementary feature subspaces for each estimator."""
        rng = np.random.RandomState(self.random_state)
        
        # Determine subset size
        if isinstance(self.feature_subset_size, float):
            subset_size = max(1, int(self.feature_subset_size * n_features))
        else:
            subset_size = min(self.feature_subset_size, n_features)
        
        feature_subspaces = []
        
        for i in range(self.n_estimators):
            # Randomly sample features without replacement
            features = rng.choice(n_features, size=subset_size, replace=False)
            feature_subspaces.append(np.sort(features))
        
        return feature_subspaces
    
    def _compute_centroids(self, X, y, feature_indices):
        """Compute class centroids for a given feature subspace."""
        centroids = {}
        
        for class_label in self.classes_:
            class_mask = (y == class_label)
            class_samples = X[class_mask][:, feature_indices]
            centroids[class_label] = np.mean(class_samples, axis=0)
        
        return centroids
    
    def fit(self, X_train, y_train):
        """
        Fit the ensemble of centroid classifiers.
        
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
        
        # Store classes
        self.classes_ = unique_labels(y_train)
        self.n_features_in_ = X_train.shape[1]
        
        # Generate feature subspaces
        self.feature_subspaces_ = self._generate_feature_subspaces(
            self.n_features_in_
        )
        
        # Train centroid classifiers on each subspace
        self.centroids_ = []
        
        for feature_indices in self.feature_subspaces_:
            centroids = self._compute_centroids(X_train, y_train, feature_indices)
            self.centroids_.append(centroids)
        
        return self
    
    def _predict_single_estimator(self, X, estimator_idx):
        """
        Compute distances to centroids for a single estimator.
        
        Returns
        -------
        distances : ndarray of shape (n_samples, n_classes)
            Distance from each sample to each class centroid.
        """
        feature_indices = self.feature_subspaces_[estimator_idx]
        centroids = self.centroids_[estimator_idx]
        
        X_subset = X[:, feature_indices]
        
        # Compute distances to each centroid
        distances = np.zeros((X.shape[0], len(self.classes_)))
        
        for class_idx, class_label in enumerate(self.classes_):
            centroid = centroids[class_label].reshape(1, -1)
            dist = pairwise_distances(
                X_subset, centroid, metric=self.metric
            ).ravel()
            distances[:, class_idx] = dist
        
        return distances
    
    def predict(self, X_test):
        """
        Predict class labels using additive ensemble of distances.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Validate input
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_test.shape[1]} features, "
                f"but the model was trained with {self.n_features_in_} features."
            )
        
        # Accumulate distances from all estimators
        total_distances = np.zeros((X_test.shape[0], len(self.classes_)))
        
        for estimator_idx in range(self.n_estimators):
            distances = self._predict_single_estimator(X_test, estimator_idx)
            total_distances += distances
        
        # Predict class with minimum total distance
        predicted_indices = np.argmin(total_distances, axis=1)
        y_pred = self.classes_[predicted_indices]
        
        return y_pred
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities based on inverse distances.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        # Accumulate distances from all estimators
        total_distances = np.zeros((X_test.shape[0], len(self.classes_)))
        
        for estimator_idx in range(self.n_estimators):
            distances = self._predict_single_estimator(X_test, estimator_idx)
            total_distances += distances
        
        # Convert distances to probabilities using inverse distances
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        inverse_distances = 1.0 / (total_distances + epsilon)
        proba = inverse_distances / inverse_distances.sum(axis=1, keepdims=True)
        
        return proba