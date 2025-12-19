import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import zlib
import bz2
import lzma
from collections import Counter


class GeometricEnsembleCompressionClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble compression classifier using multiple algorithmic information measures
    weighted by their geometric distance in information space.
    
    This classifier uses multiple compression algorithms (zlib, bz2, lzma) to compute
    normalized compression distances (NCD) between samples. The predictions from each
    compression method are weighted based on their geometric distances in the 
    information space.
    
    Parameters
    ----------
    n_neighbors : int, default=3
        Number of nearest neighbors to consider for classification.
    
    distance_metric : str, default='euclidean'
        Metric to compute geometric distances in information space.
        
    normalize : bool, default=True
        Whether to normalize the compression distances.
    """
    
    def __init__(self, n_neighbors=3, distance_metric='euclidean', normalize=True):
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.normalize = normalize
        
    def _compress_zlib(self, data):
        """Compress data using zlib."""
        return len(zlib.compress(data.tobytes(), level=9))
    
    def _compress_bz2(self, data):
        """Compress data using bz2."""
        return len(bz2.compress(data.tobytes(), compresslevel=9))
    
    def _compress_lzma(self, data):
        """Compress data using lzma."""
        return len(lzma.compress(data.tobytes(), preset=9))
    
    def _normalized_compression_distance(self, x1, x2, compress_func):
        """
        Compute normalized compression distance between two samples.
        
        NCD(x,y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
        """
        c_x = compress_func(x1)
        c_y = compress_func(x2)
        
        # Concatenate and compress
        xy = np.concatenate([x1, x2])
        c_xy = compress_func(xy)
        
        ncd = (c_xy - min(c_x, c_y)) / max(c_x, c_y)
        return max(0, ncd)  # Ensure non-negative
    
    def _compute_distance_matrix(self, X1, X2, compress_func):
        """Compute pairwise NCD distance matrix between samples."""
        n1, n2 = len(X1), len(X2)
        distances = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                distances[i, j] = self._normalized_compression_distance(
                    X1[i], X2[j], compress_func
                )
        
        return distances
    
    def _compute_information_space_embedding(self, X):
        """
        Compute embedding in information space using compression ratios.
        Returns a feature vector for each sample based on compression characteristics.
        """
        n_samples = len(X)
        embeddings = np.zeros((n_samples, 3))
        
        for i in range(n_samples):
            sample = X[i]
            # Compression ratios as features
            embeddings[i, 0] = self._compress_zlib(sample) / sample.nbytes
            embeddings[i, 1] = self._compress_bz2(sample) / sample.nbytes
            embeddings[i, 2] = self._compress_lzma(sample) / sample.nbytes
            
        return embeddings
    
    def _compute_geometric_weights(self, embeddings):
        """
        Compute weights based on geometric distances in information space.
        Methods that are closer in information space get higher weights.
        """
        # Compute pairwise distances between compression methods
        # Each column represents a compression method's behavior across samples
        method_vectors = embeddings.T  # Shape: (3, n_samples)
        
        # Compute correlation-based distances between methods
        correlations = np.corrcoef(method_vectors)
        
        # Convert correlations to distances
        distances = 1 - np.abs(correlations)
        
        # Compute weights inversely proportional to average distance
        avg_distances = distances.mean(axis=1)
        
        # Inverse distance weighting with smoothing
        weights = 1.0 / (avg_distances + 1e-6)
        weights = weights / weights.sum()  # Normalize
        
        return weights
    
    def _predict_single_method(self, distances, y_train):
        """Predict using k-nearest neighbors for a single compression method."""
        n_test = distances.shape[0]
        predictions = np.zeros(n_test, dtype=int)
        
        for i in range(n_test):
            # Get k nearest neighbors
            nearest_indices = np.argsort(distances[i])[:self.n_neighbors]
            nearest_labels = y_train[nearest_indices]
            
            # Majority vote
            label_counts = Counter(nearest_labels)
            predictions[i] = label_counts.most_common(1)[0][0]
        
        return predictions
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier.
        
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
        
        # Store training data
        self.X_train_ = X_train.copy()
        self.y_train_ = y_train.copy()
        
        # Compute information space embeddings for training data
        self.train_embeddings_ = self._compute_information_space_embedding(X_train)
        
        # Compute geometric weights based on training data
        self.weights_ = self._compute_geometric_weights(self.train_embeddings_)
        
        # Store compression functions
        self.compress_funcs_ = [
            self._compress_zlib,
            self._compress_bz2,
            self._compress_lzma
        ]
        
        return self
    
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
        # Check if fitted
        check_is_fitted(self, ['X_train_', 'y_train_', 'weights_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        n_test = len(X_test)
        n_methods = len(self.compress_funcs_)
        
        # Store predictions from each method
        all_predictions = np.zeros((n_test, n_methods), dtype=int)
        
        # Get predictions from each compression method
        for method_idx, compress_func in enumerate(self.compress_funcs_):
            # Compute distance matrix
            distances = self._compute_distance_matrix(
                X_test, self.X_train_, compress_func
            )
            
            # Get predictions
            all_predictions[:, method_idx] = self._predict_single_method(
                distances, self.y_train_
            )
        
        # Weighted voting
        final_predictions = np.zeros(n_test, dtype=int)
        
        for i in range(n_test):
            # Count weighted votes for each class
            class_votes = {}
            for class_label in self.classes_:
                class_votes[class_label] = 0.0
            
            for method_idx in range(n_methods):
                predicted_class = all_predictions[i, method_idx]
                class_votes[predicted_class] += self.weights_[method_idx]
            
            # Select class with highest weighted vote
            final_predictions[i] = max(class_votes, key=class_votes.get)
        
        return final_predictions
    
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
        check_is_fitted(self, ['X_train_', 'y_train_', 'weights_'])
        X_test = check_array(X_test)
        
        n_test = len(X_test)
        n_classes = len(self.classes_)
        n_methods = len(self.compress_funcs_)
        
        proba = np.zeros((n_test, n_classes))
        
        # Get predictions from each method
        for method_idx, compress_func in enumerate(self.compress_funcs_):
            distances = self._compute_distance_matrix(
                X_test, self.X_train_, compress_func
            )
            
            for i in range(n_test):
                # Get k nearest neighbors
                nearest_indices = np.argsort(distances[i])[:self.n_neighbors]
                nearest_labels = self.y_train_[nearest_indices]
                
                # Count votes for each class
                for class_idx, class_label in enumerate(self.classes_):
                    count = np.sum(nearest_labels == class_label)
                    proba[i, class_idx] += (count / self.n_neighbors) * self.weights_[method_idx]
        
        # Normalize probabilities
        proba = proba / proba.sum(axis=1, keepdims=True)
        
        return proba