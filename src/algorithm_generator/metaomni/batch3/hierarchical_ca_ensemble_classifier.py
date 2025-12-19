import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import cdist
from typing import List, Tuple, Callable


class HierarchicalCAEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Hierarchical Cellular Automata Ensemble Classifier.
    
    Each level compresses representations from the previous level using different
    neighborhood topologies (von Neumann, Moore, extended neighborhoods).
    
    Parameters
    ----------
    n_levels : int, default=3
        Number of hierarchical levels in the ensemble.
    n_iterations : int, default=5
        Number of CA iterations per level.
    compression_factor : float, default=0.5
        Factor by which to compress representations at each level.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_levels=3, n_iterations=5, compression_factor=0.5, random_state=None):
        self.n_levels = n_levels
        self.n_iterations = n_iterations
        self.compression_factor = compression_factor
        self.random_state = random_state
        
    def _get_neighborhood_topology(self, level):
        """Get neighborhood topology function for a given level."""
        topologies = [
            self._von_neumann_neighbors,
            self._moore_neighbors,
            self._extended_neighbors
        ]
        return topologies[level % len(topologies)]
    
    def _von_neumann_neighbors(self, idx, grid_shape):
        """Von Neumann neighborhood (4-connectivity in 2D)."""
        neighbors = []
        row, col = idx // grid_shape[1], idx % grid_shape[1]
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]:
                neighbors.append(nr * grid_shape[1] + nc)
        return neighbors
    
    def _moore_neighbors(self, idx, grid_shape):
        """Moore neighborhood (8-connectivity in 2D)."""
        neighbors = []
        row, col = idx // grid_shape[1], idx % grid_shape[1]
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]:
                    neighbors.append(nr * grid_shape[1] + nc)
        return neighbors
    
    def _extended_neighbors(self, idx, grid_shape):
        """Extended neighborhood (distance-2 connectivity)."""
        neighbors = []
        row, col = idx // grid_shape[1], idx % grid_shape[1]
        
        for dr in [-2, -1, 0, 1, 2]:
            for dc in [-2, -1, 0, 1, 2]:
                if dr == 0 and dc == 0:
                    continue
                if abs(dr) + abs(dc) <= 3:  # Manhattan distance constraint
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < grid_shape[0] and 0 <= nc < grid_shape[1]:
                        neighbors.append(nr * grid_shape[1] + nc)
        return neighbors
    
    def _create_grid(self, X):
        """Create a 2D grid from feature vectors."""
        n_samples, n_features = X.shape
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        grid_shape = (grid_size, grid_size)
        
        # Pad if necessary
        padded_size = grid_size * grid_size
        if padded_size > n_samples:
            padding = np.zeros((padded_size - n_samples, n_features))
            X_padded = np.vstack([X, padding])
        else:
            X_padded = X
            
        return X_padded, grid_shape
    
    def _ca_update(self, state, grid_shape, topology_func, weights):
        """Perform one CA update step."""
        new_state = np.zeros_like(state)
        
        for idx in range(len(state)):
            neighbors = topology_func(idx, grid_shape)
            if len(neighbors) > 0:
                # Filter out neighbors that are out of bounds
                valid_neighbors = [n for n in neighbors if n < len(state)]
                if len(valid_neighbors) > 0:
                    neighbor_states = state[valid_neighbors]
                    # Weighted aggregation of neighbor states
                    new_state[idx] = np.tanh(
                        state[idx] + np.mean(neighbor_states @ weights.T, axis=0)
                    )
                else:
                    new_state[idx] = state[idx]
            else:
                new_state[idx] = state[idx]
                
        return new_state
    
    def _compress_representation(self, X, target_dim, projection_matrix=None):
        """Compress representation using random projection."""
        if projection_matrix is None:
            rng = np.random.RandomState(self.random_state)
            projection_matrix = rng.randn(X.shape[1], target_dim) / np.sqrt(target_dim)
        return X @ projection_matrix, projection_matrix
    
    def _train_level(self, X, y, level):
        """Train a single CA level."""
        X_grid, grid_shape = self._create_grid(X)
        n_features = X_grid.shape[1]
        
        # Initialize CA weights
        rng = np.random.RandomState(self.random_state + level if self.random_state else None)
        weights = rng.randn(n_features, n_features) * 0.1
        
        # Get topology for this level
        topology_func = self._get_neighborhood_topology(level)
        
        # Run CA iterations
        state = X_grid.copy()
        for _ in range(self.n_iterations):
            state = self._ca_update(state, grid_shape, topology_func, weights)
        
        # Extract features for classification
        features = state[:len(X)]
        
        # Train simple classifier on CA-transformed features
        class_prototypes = {}
        for class_label in np.unique(y):
            class_mask = y == class_label
            class_prototypes[class_label] = np.mean(features[class_mask], axis=0)
        
        return {
            'weights': weights,
            'topology_func': topology_func,
            'prototypes': class_prototypes,
            'features': features,
            'n_features': n_features
        }
    
    def fit(self, X, y):
        """
        Fit the hierarchical CA ensemble classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        rng = np.random.RandomState(self.random_state)
        
        # Train hierarchical levels
        self.levels_ = []
        self.projection_matrices_ = []
        X_current = X.copy()
        
        for level in range(self.n_levels):
            level_data = self._train_level(X_current, y, level)
            self.levels_.append(level_data)
            
            # Compress for next level
            if level < self.n_levels - 1:
                target_dim = max(2, int(X_current.shape[1] * self.compression_factor))
                X_current, proj_matrix = self._compress_representation(
                    level_data['features'], target_dim
                )
                self.projection_matrices_.append(proj_matrix)
            else:
                self.projection_matrices_.append(None)
        
        self.X_ = X
        self.y_ = y
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # Collect predictions from all levels
        all_predictions = []
        X_current = X.copy()
        
        for level, level_data in enumerate(self.levels_):
            # Transform through CA - use current grid shape for test data
            X_grid, grid_shape = self._create_grid(X_current)
            state = X_grid.copy()
            
            for _ in range(self.n_iterations):
                state = self._ca_update(
                    state, 
                    grid_shape,  # Use the actual grid shape for this data
                    level_data['topology_func'], 
                    level_data['weights']
                )
            
            features = state[:len(X)]
            
            # Predict using prototypes
            predictions = np.zeros(len(X), dtype=int)
            for i, feature in enumerate(features):
                distances = {}
                for class_label, prototype in level_data['prototypes'].items():
                    distances[class_label] = np.linalg.norm(feature - prototype)
                predictions[i] = min(distances, key=distances.get)
            
            all_predictions.append(predictions)
            
            # Compress for next level using stored projection matrix
            if level < self.n_levels - 1 and self.projection_matrices_[level] is not None:
                X_current, _ = self._compress_representation(
                    features, 
                    self.projection_matrices_[level].shape[1],
                    self.projection_matrices_[level]
                )
        
        # Ensemble voting
        all_predictions = np.array(all_predictions)
        final_predictions = np.zeros(len(X), dtype=int)
        
        for i in range(len(X)):
            votes = np.bincount(all_predictions[:, i], minlength=self.n_classes_)
            final_predictions[i] = self.classes_[np.argmax(votes)]
        
        return final_predictions