import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mode
from collections import defaultdict


class HierarchicalFractalDimensionClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses hierarchical fractal dimension estimation across
    multiple compression scales to capture multi-resolution patterns in the
    data manifold.
    
    Parameters
    ----------
    n_scales : int, default=5
        Number of hierarchical scales to use for fractal dimension estimation.
    
    min_radius : float, default=0.01
        Minimum radius for box-counting algorithm.
    
    max_radius : float, default=1.0
        Maximum radius for box-counting algorithm.
    
    n_radii : int, default=20
        Number of radii to sample between min_radius and max_radius.
    
    k_neighbors : int, default=5
        Number of neighbors to consider for local fractal dimension estimation.
    
    metric : str, default='euclidean'
        Distance metric to use.
    
    aggregation : str, default='mean'
        How to aggregate fractal dimensions across scales ('mean', 'concat', 'weighted').
    """
    
    def __init__(self, n_scales=5, min_radius=0.01, max_radius=1.0, 
                 n_radii=20, k_neighbors=5, metric='euclidean', 
                 aggregation='mean'):
        self.n_scales = n_scales
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.n_radii = n_radii
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.aggregation = aggregation
    
    def _estimate_local_fractal_dimension(self, X, point_idx, radii):
        """
        Estimate local fractal dimension around a point using correlation dimension.
        """
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        distances = distances[distances > 0]  # Remove self-distance
        
        if len(distances) < 2:
            return 1.0
        
        counts = []
        valid_radii = []
        
        for r in radii:
            count = np.sum(distances <= r)
            if count > 0:
                counts.append(count)
                valid_radii.append(r)
        
        if len(counts) < 2:
            return 1.0
        
        # Linear regression in log-log space
        log_r = np.log(valid_radii)
        log_c = np.log(counts)
        
        # Avoid numerical issues
        valid_mask = np.isfinite(log_r) & np.isfinite(log_c)
        if np.sum(valid_mask) < 2:
            return 1.0
        
        log_r = log_r[valid_mask]
        log_c = log_c[valid_mask]
        
        # Fit line: log(C) = D * log(r) + b
        coeffs = np.polyfit(log_r, log_c, 1)
        fractal_dim = coeffs[0]
        
        # Clamp to reasonable range
        return np.clip(fractal_dim, 0.1, X.shape[1] + 1)
    
    def _compute_hierarchical_features(self, X):
        """
        Compute hierarchical fractal dimension features across multiple scales.
        """
        n_samples = X.shape[0]
        features_per_scale = []
        
        # Generate radii for box-counting
        radii = np.logspace(np.log10(self.min_radius), 
                           np.log10(self.max_radius), 
                           self.n_radii)
        
        for scale_idx in range(self.n_scales):
            # Subsample data at different scales (hierarchical compression)
            compression_factor = 2 ** scale_idx
            step = max(1, n_samples // (n_samples // compression_factor + 1))
            indices = np.arange(0, n_samples, step)
            X_compressed = X[indices]
            
            # Compute local fractal dimensions
            scale_features = []
            for i in range(len(X_compressed)):
                local_fd = self._estimate_local_fractal_dimension(
                    X_compressed, i, radii
                )
                scale_features.append(local_fd)
            
            # Aggregate features at this scale
            scale_features = np.array(scale_features)
            
            # Compute statistics of fractal dimensions at this scale
            feature_vector = [
                np.mean(scale_features),
                np.std(scale_features),
                np.median(scale_features),
                np.percentile(scale_features, 25),
                np.percentile(scale_features, 75),
            ]
            
            features_per_scale.append(feature_vector)
        
        # Aggregate across scales
        if self.aggregation == 'concat':
            return np.concatenate(features_per_scale)
        elif self.aggregation == 'mean':
            return np.mean(features_per_scale, axis=0)
        elif self.aggregation == 'weighted':
            # Weight by scale (finer scales get more weight)
            weights = np.array([1.0 / (2 ** i) for i in range(self.n_scales)])
            weights = weights / weights.sum()
            return np.average(features_per_scale, axis=0, weights=weights)
        else:
            return np.mean(features_per_scale, axis=0)
    
    def _compute_global_fractal_dimension(self, X):
        """
        Compute global fractal dimension using box-counting method.
        """
        # Normalize data to unit hypercube
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-10)
        
        # Generate box sizes
        box_sizes = np.logspace(-2, 0, 15)
        counts = []
        
        for box_size in box_sizes:
            # Count occupied boxes
            n_boxes_per_dim = int(1.0 / box_size) + 1
            box_indices = (X_norm / box_size).astype(int)
            box_indices = np.clip(box_indices, 0, n_boxes_per_dim - 1)
            
            # Count unique boxes
            unique_boxes = set(map(tuple, box_indices))
            counts.append(len(unique_boxes))
        
        # Fit power law
        log_sizes = np.log(box_sizes)
        log_counts = np.log(counts)
        
        valid_mask = np.isfinite(log_sizes) & np.isfinite(log_counts)
        if np.sum(valid_mask) < 2:
            return X.shape[1]
        
        coeffs = np.polyfit(log_sizes[valid_mask], log_counts[valid_mask], 1)
        return -coeffs[0]  # Negative slope is the fractal dimension
    
    def fit(self, X_train, y_train):
        """
        Fit the hierarchical fractal dimension classifier.
        
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
        X_train, y_train = check_X_y(X_train, y_train)
        self.classes_ = unique_labels(y_train)
        self.n_features_in_ = X_train.shape[1]
        
        # Store class-specific fractal signatures
        self.class_signatures_ = {}
        self.class_samples_ = {}
        
        for cls in self.classes_:
            X_cls = X_train[y_train == cls]
            self.class_samples_[cls] = X_cls
            
            # Compute hierarchical fractal features for this class
            hierarchical_features = self._compute_hierarchical_features(X_cls)
            
            # Compute global fractal dimension
            global_fd = self._compute_global_fractal_dimension(X_cls)
            
            # Combine features
            signature = np.concatenate([
                hierarchical_features,
                [global_fd]
            ])
            
            self.class_signatures_[cls] = signature
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_test.shape[1]}")
        
        predictions = []
        
        for x in X_test:
            x_reshaped = x.reshape(1, -1)
            
            # Compute distances to each class based on fractal signatures
            distances = {}
            
            for cls in self.classes_:
                # Combine test point with class samples
                X_combined = np.vstack([self.class_samples_[cls], x_reshaped])
                
                # Compute hierarchical features
                hierarchical_features = self._compute_hierarchical_features(X_combined)
                global_fd = self._compute_global_fractal_dimension(X_combined)
                
                test_signature = np.concatenate([
                    hierarchical_features,
                    [global_fd]
                ])
                
                # Compute distance to class signature
                distance = np.linalg.norm(test_signature - self.class_signatures_[cls])
                distances[cls] = distance
            
            # Predict class with minimum distance
            predicted_class = min(distances, key=distances.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        probas = []
        
        for x in X_test:
            x_reshaped = x.reshape(1, -1)
            
            distances = {}
            
            for cls in self.classes_:
                X_combined = np.vstack([self.class_samples_[cls], x_reshaped])
                hierarchical_features = self._compute_hierarchical_features(X_combined)
                global_fd = self._compute_global_fractal_dimension(X_combined)
                
                test_signature = np.concatenate([
                    hierarchical_features,
                    [global_fd]
                ])
                
                distance = np.linalg.norm(test_signature - self.class_signatures_[cls])
                distances[cls] = distance
            
            # Convert distances to probabilities using softmax
            distance_array = np.array([distances[cls] for cls in self.classes_])
            # Use negative distances for softmax (smaller distance = higher probability)
            exp_neg_dist = np.exp(-distance_array)
            proba = exp_neg_dist / exp_neg_dist.sum()
            
            probas.append(proba)
        
        return np.array(probas)