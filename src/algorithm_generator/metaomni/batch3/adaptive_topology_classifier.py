import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import cdist
from scipy.interpolate import Rbf


class AdaptiveTopologyClassifier(BaseEstimator, ClassifierMixin):
    """
    Dynamic topology compression classifier that adapts map dimensions based on
    predictive certainty. Regions with high certainty are compressed while
    ambiguous decision boundaries are expanded for better resolution.
    
    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors for uncertainty estimation
    compression_factor : float, default=2.0
        Factor controlling compression strength in certain regions
    expansion_factor : float, default=2.0
        Factor controlling expansion strength in uncertain regions
    certainty_threshold : float, default=0.7
        Threshold for determining high certainty regions
    base_estimator : str, default='knn'
        Base classifier to use ('knn' or 'tree')
    n_grid_points : int, default=50
        Number of grid points per dimension for topology mapping
    """
    
    def __init__(self, n_neighbors=5, compression_factor=2.0, 
                 expansion_factor=2.0, certainty_threshold=0.7,
                 base_estimator='knn', n_grid_points=50):
        self.n_neighbors = n_neighbors
        self.compression_factor = compression_factor
        self.expansion_factor = expansion_factor
        self.certainty_threshold = certainty_threshold
        self.base_estimator = base_estimator
        self.n_grid_points = n_grid_points
    
    def _compute_local_certainty(self, X):
        """
        Compute local predictive certainty for each point using k-NN agreement.
        """
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        knn.fit(self.X_train_, self.y_train_)
        
        # Get k nearest neighbors
        distances, indices = knn.kneighbors(X)
        
        # Compute certainty as agreement among neighbors
        certainty = np.zeros(len(X))
        for i in range(len(X)):
            neighbor_labels = self.y_train_[indices[i]]
            # Certainty is the proportion of the most common class
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            certainty[i] = np.max(counts) / len(neighbor_labels)
        
        return certainty
    
    def _create_topology_map(self):
        """
        Create a topology transformation map based on local certainty.
        """
        # Create grid over feature space
        n_features = self.X_train_.shape[1]
        grid_axes = []
        
        for dim in range(n_features):
            min_val = self.X_train_[:, dim].min()
            max_val = self.X_train_[:, dim].max()
            margin = (max_val - min_val) * 0.1
            grid_axes.append(np.linspace(min_val - margin, max_val + margin, 
                                        self.n_grid_points))
        
        # Create meshgrid
        if n_features == 1:
            grid_points = grid_axes[0].reshape(-1, 1)
        else:
            meshes = np.meshgrid(*grid_axes, indexing='ij')
            grid_points = np.column_stack([m.ravel() for m in meshes])
        
        # Compute certainty at grid points
        certainty = self._compute_local_certainty(grid_points)
        
        # Store grid information
        self.grid_points_ = grid_points
        self.grid_certainty_ = certainty
        self.grid_axes_ = grid_axes
        
        return certainty
    
    def _transform_space(self, X, inverse=False):
        """
        Transform feature space based on local certainty.
        High certainty regions are compressed, uncertain regions are expanded.
        """
        if not hasattr(self, 'grid_certainty_'):
            return X
        
        X_transformed = X.copy()
        
        # For each point, find local certainty and apply transformation
        for i in range(len(X)):
            point = X[i:i+1]
            
            # Find nearest grid points to estimate local certainty
            distances = cdist(point, self.grid_points_)
            nearest_idx = np.argmin(distances, axis=1)[0]
            local_certainty = self.grid_certainty_[nearest_idx]
            
            # Compute transformation factor
            if local_certainty > self.certainty_threshold:
                # High certainty: compress (move toward local mean)
                factor = 1.0 / self.compression_factor
            else:
                # Low certainty: expand (move away from local mean)
                factor = self.expansion_factor
            
            # Find local neighborhood center
            neighbor_distances = cdist(point, self.X_train_)
            k_nearest = np.argsort(neighbor_distances[0])[:self.n_neighbors]
            local_center = self.X_train_[k_nearest].mean(axis=0)
            
            # Apply transformation
            if not inverse:
                direction = point[0] - local_center
                X_transformed[i] = local_center + direction * factor
            else:
                direction = point[0] - local_center
                X_transformed[i] = local_center + direction / factor
        
        return X_transformed
    
    def fit(self, X, y):
        """
        Fit the adaptive topology classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Validate input
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Store training data
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        
        # Create topology map based on certainty
        self._create_topology_map()
        
        # Transform training data
        X_transformed = self._transform_space(X)
        
        # Fit base classifier on transformed space
        if self.base_estimator == 'knn':
            self.classifier_ = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        else:
            self.classifier_ = DecisionTreeClassifier(max_depth=10, 
                                                     min_samples_split=5)
        
        self.classifier_.fit(X_transformed, y)
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        # Check if fitted
        check_is_fitted(self, ['classifier_', 'X_train_', 'y_train_'])
        
        # Validate input
        X = check_array(X)
        
        # Transform test data using the same topology
        X_transformed = self._transform_space(X)
        
        # Predict using base classifier
        y_pred = self.classifier_.predict(X_transformed)
        
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self, ['classifier_', 'X_train_', 'y_train_'])
        X = check_array(X)
        
        X_transformed = self._transform_space(X)
        
        if hasattr(self.classifier_, 'predict_proba'):
            return self.classifier_.predict_proba(X_transformed)
        else:
            # Fallback for classifiers without predict_proba
            predictions = self.classifier_.predict(X_transformed)
            n_classes = len(self.classes_)
            proba = np.zeros((len(X), n_classes))
            for i, pred in enumerate(predictions):
                class_idx = np.where(self.classes_ == pred)[0][0]
                proba[i, class_idx] = 1.0
            return proba