import numpy as np
import zlib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from collections import defaultdict


class ProgressiveKolmogorovComplexityClassifier(BaseEstimator, ClassifierMixin):
    """
    Progressive Refinement Kolmogorov Complexity Classifier.
    
    Uses coarse compression to identify high-density regions, then applies
    fine-grained local Kolmogorov complexity estimation for classification.
    
    Parameters
    ----------
    n_coarse_clusters : int, default=10
        Number of clusters for initial coarse partitioning
    density_threshold : float, default=0.7
        Threshold percentile for identifying high-density regions
    refinement_levels : int, default=3
        Number of progressive refinement levels
    compression_method : str, default='zlib'
        Compression method for complexity estimation
    n_neighbors : int, default=5
        Number of neighbors for local complexity estimation
    random_state : int, default=None
        Random state for reproducibility
    """
    
    def __init__(self, n_coarse_clusters=10, density_threshold=0.7,
                 refinement_levels=3, compression_method='zlib',
                 n_neighbors=5, random_state=None):
        self.n_coarse_clusters = n_coarse_clusters
        self.density_threshold = density_threshold
        self.refinement_levels = refinement_levels
        self.compression_method = compression_method
        self.n_neighbors = n_neighbors
        self.random_state = random_state
    
    def _compress(self, data):
        """Estimate Kolmogorov complexity via compression."""
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif isinstance(data, (list, tuple)):
            data = np.array(data).tobytes()
        
        if self.compression_method == 'zlib':
            return len(zlib.compress(data, level=9))
        else:
            return len(zlib.compress(data, level=9))
    
    def _normalized_compression_distance(self, x1, x2):
        """Compute normalized compression distance between two samples."""
        c_x1 = self._compress(x1)
        c_x2 = self._compress(x2)
        c_x1x2 = self._compress(np.concatenate([x1.flatten(), x2.flatten()]))
        
        ncd = (c_x1x2 - min(c_x1, c_x2)) / max(c_x1, c_x2)
        return max(0, min(1, ncd))
    
    def _compute_local_complexity(self, sample, reference_samples):
        """Compute local Kolmogorov complexity in a region."""
        complexities = []
        for ref in reference_samples:
            ncd = self._normalized_compression_distance(sample, ref)
            complexities.append(ncd)
        return np.mean(complexities) if complexities else 0.0
    
    def _identify_high_density_regions(self, X, y):
        """Identify high-density regions using coarse clustering."""
        kmeans = KMeans(n_clusters=self.n_coarse_clusters, 
                       random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Compute density for each cluster
        cluster_densities = []
        for i in range(self.n_coarse_clusters):
            mask = cluster_labels == i
            if np.sum(mask) > 0:
                cluster_points = X[mask]
                center = kmeans.cluster_centers_[i]
                avg_dist = np.mean(np.linalg.norm(cluster_points - center, axis=1))
                density = np.sum(mask) / (avg_dist + 1e-10)
                cluster_densities.append(density)
            else:
                cluster_densities.append(0)
        
        # Identify high-density clusters
        density_threshold_val = np.percentile(cluster_densities, 
                                             self.density_threshold * 100)
        high_density_clusters = [i for i, d in enumerate(cluster_densities) 
                                if d >= density_threshold_val]
        
        return cluster_labels, high_density_clusters, kmeans
    
    def _progressive_refinement(self, X, y, cluster_labels, high_density_clusters):
        """Apply progressive refinement in high-density regions."""
        refined_prototypes = defaultdict(list)
        
        for class_label in self.classes_:
            class_mask = y == class_label
            
            for level in range(self.refinement_levels):
                for cluster_id in high_density_clusters:
                    # Get samples in this cluster and class
                    region_mask = (cluster_labels == cluster_id) & class_mask
                    region_samples = X[region_mask]
                    
                    if len(region_samples) == 0:
                        continue
                    
                    # Progressive refinement: subsample at each level
                    n_samples = max(1, len(region_samples) // (2 ** level))
                    if n_samples < len(region_samples):
                        indices = np.random.choice(len(region_samples), 
                                                  n_samples, replace=False)
                        region_samples = region_samples[indices]
                    
                    # Compute complexity-based prototypes
                    for sample in region_samples:
                        complexity = self._compute_local_complexity(
                            sample, region_samples[:min(self.n_neighbors, len(region_samples))]
                        )
                        refined_prototypes[class_label].append({
                            'sample': sample,
                            'complexity': complexity,
                            'level': level,
                            'cluster': cluster_id
                        })
        
        return refined_prototypes
    
    def fit(self, X, y):
        """
        Fit the Progressive Kolmogorov Complexity Classifier.
        
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
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Step 1: Coarse compression - identify high-density regions
        cluster_labels, high_density_clusters, kmeans = \
            self._identify_high_density_regions(X, y)
        
        self.coarse_kmeans_ = kmeans
        self.high_density_clusters_ = high_density_clusters
        
        # Step 2: Progressive refinement in high-density regions
        self.refined_prototypes_ = self._progressive_refinement(
            X, y, cluster_labels, high_density_clusters
        )
        
        # Step 3: Store coarse prototypes for low-density regions
        self.coarse_prototypes_ = {}
        for class_label in self.classes_:
            class_mask = y == class_label
            class_samples = X[class_mask]
            
            # Store representative samples
            n_representatives = min(20, len(class_samples))
            if n_representatives > 0:
                indices = np.random.choice(len(class_samples), 
                                         n_representatives, replace=False)
                self.coarse_prototypes_[class_label] = class_samples[indices]
        
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
        check_is_fitted(self)
        X = check_array(X)
        
        predictions = []
        
        for sample in X:
            # Determine if sample is in high-density region
            cluster_id = self.coarse_kmeans_.predict(sample.reshape(1, -1))[0]
            is_high_density = cluster_id in self.high_density_clusters_
            
            class_scores = {}
            
            for class_label in self.classes_:
                if is_high_density and class_label in self.refined_prototypes_:
                    # Use fine-grained complexity estimation
                    prototypes = self.refined_prototypes_[class_label]
                    relevant_prototypes = [p for p in prototypes 
                                         if p['cluster'] == cluster_id]
                    
                    if relevant_prototypes:
                        complexities = []
                        for proto in relevant_prototypes[:self.n_neighbors]:
                            ncd = self._normalized_compression_distance(
                                sample, proto['sample']
                            )
                            # Weight by refinement level (finer = more weight)
                            weight = 1.0 / (proto['level'] + 1)
                            complexities.append(ncd * weight)
                        
                        class_scores[class_label] = -np.mean(complexities)
                    else:
                        # Fallback to coarse prototypes
                        if class_label in self.coarse_prototypes_:
                            prototypes = self.coarse_prototypes_[class_label]
                            distances = cdist(sample.reshape(1, -1), prototypes)
                            class_scores[class_label] = -np.min(distances)
                        else:
                            class_scores[class_label] = -np.inf
                else:
                    # Use coarse prototypes for low-density regions
                    if class_label in self.coarse_prototypes_:
                        prototypes = self.coarse_prototypes_[class_label]
                        distances = cdist(sample.reshape(1, -1), prototypes)
                        class_scores[class_label] = -np.min(distances)
                    else:
                        class_scores[class_label] = -np.inf
            
            # Predict class with highest score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)