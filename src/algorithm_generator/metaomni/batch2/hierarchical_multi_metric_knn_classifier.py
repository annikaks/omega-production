import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from collections import Counter


class HierarchicalMultiMetricKNNClassifier(BaseEstimator, ClassifierMixin):
    """
    A hierarchical KNN classifier that uses multiple distance metrics at different granularities.
    
    First performs coarse clustering to identify relevant regions, then applies fine-grained
    local distance refinement for final classification.
    
    Parameters
    ----------
    n_clusters : int, default=10
        Number of coarse clusters to create
    n_neighbors : int, default=5
        Number of neighbors to use for final classification
    coarse_metric : str, default='euclidean'
        Distance metric for coarse clustering ('euclidean', 'manhattan', 'cosine')
    fine_metric : str, default='euclidean'
        Distance metric for fine-grained neighbor search
    cluster_weight : float, default=0.3
        Weight for cluster membership in final distance calculation (0-1)
    scale_features : bool, default=True
        Whether to standardize features before processing
    """
    
    def __init__(self, n_clusters=10, n_neighbors=5, coarse_metric='euclidean',
                 fine_metric='euclidean', cluster_weight=0.3, scale_features=True):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.coarse_metric = coarse_metric
        self.fine_metric = fine_metric
        self.cluster_weight = cluster_weight
        self.scale_features = scale_features
        
    def _compute_distance(self, X1, X2, metric):
        """Compute pairwise distances using specified metric."""
        if metric == 'euclidean':
            return euclidean_distances(X1, X2)
        elif metric == 'manhattan':
            return manhattan_distances(X1, X2)
        elif metric == 'cosine':
            return cosine_distances(X1, X2)
        else:
            return cdist(X1, X2, metric=metric)
    
    def fit(self, X_train, y_train):
        """
        Fit the hierarchical multi-metric classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Returns self
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        # Store training data
        self.X_train_ = X_train.copy()
        self.y_train_ = y_train.copy()
        self.classes_ = np.unique(y_train)
        
        # Scale features if requested
        if self.scale_features:
            self.scaler_ = StandardScaler()
            self.X_train_scaled_ = self.scaler_.fit_transform(X_train)
        else:
            self.X_train_scaled_ = X_train
        
        # Perform coarse clustering
        n_clusters = min(self.n_clusters, len(X_train))
        self.kmeans_ = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.train_clusters_ = self.kmeans_.fit_predict(self.X_train_scaled_)
        
        # Store cluster information for each training point
        self.cluster_labels_ = {}
        for cluster_id in range(n_clusters):
            cluster_mask = self.train_clusters_ == cluster_id
            self.cluster_labels_[cluster_id] = {
                'indices': np.where(cluster_mask)[0],
                'X': self.X_train_scaled_[cluster_mask],
                'y': y_train[cluster_mask]
            }
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        X_test = np.asarray(X_test)
        
        # Scale test data
        if self.scale_features:
            X_test_scaled = self.scaler_.transform(X_test)
        else:
            X_test_scaled = X_test
        
        predictions = []
        
        for test_point in X_test_scaled:
            test_point = test_point.reshape(1, -1)
            
            # Step 1: Assign test point to clusters (coarse granularity)
            cluster_distances = self._compute_distance(
                test_point, 
                self.kmeans_.cluster_centers_, 
                self.coarse_metric
            )[0]
            
            # Normalize cluster distances to [0, 1]
            if cluster_distances.max() > 0:
                cluster_distances_norm = cluster_distances / cluster_distances.max()
            else:
                cluster_distances_norm = cluster_distances
            
            # Step 2: Compute fine-grained distances with cluster weighting
            all_distances = []
            all_indices = []
            
            for cluster_id, cluster_info in self.cluster_labels_.items():
                if len(cluster_info['indices']) == 0:
                    continue
                
                # Fine-grained local distances
                local_distances = self._compute_distance(
                    test_point,
                    cluster_info['X'],
                    self.fine_metric
                )[0]
                
                # Normalize local distances
                if local_distances.max() > 0:
                    local_distances_norm = local_distances / local_distances.max()
                else:
                    local_distances_norm = local_distances
                
                # Combine cluster membership weight with local distance
                cluster_weight_value = cluster_distances_norm[cluster_id]
                combined_distances = (
                    self.cluster_weight * cluster_weight_value +
                    (1 - self.cluster_weight) * local_distances_norm
                )
                
                all_distances.extend(combined_distances)
                all_indices.extend(cluster_info['indices'])
            
            # Step 3: Find k nearest neighbors based on combined distances
            all_distances = np.array(all_distances)
            all_indices = np.array(all_indices)
            
            k = min(self.n_neighbors, len(all_distances))
            nearest_indices = np.argpartition(all_distances, k-1)[:k]
            
            # Get labels of nearest neighbors
            neighbor_labels = self.y_train_[all_indices[nearest_indices]]
            
            # Step 4: Majority voting with distance weighting
            neighbor_distances = all_distances[nearest_indices]
            
            # Inverse distance weighting (avoid division by zero)
            weights = 1.0 / (neighbor_distances + 1e-10)
            
            # Weighted voting
            class_weights = {}
            for label, weight in zip(neighbor_labels, weights):
                class_weights[label] = class_weights.get(label, 0) + weight
            
            # Predict class with highest weighted vote
            predicted_class = max(class_weights.items(), key=lambda x: x[1])[0]
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Class probabilities
        """
        X_test = np.asarray(X_test)
        
        # Scale test data
        if self.scale_features:
            X_test_scaled = self.scaler_.transform(X_test)
        else:
            X_test_scaled = X_test
        
        probabilities = []
        
        for test_point in X_test_scaled:
            test_point = test_point.reshape(1, -1)
            
            # Coarse clustering
            cluster_distances = self._compute_distance(
                test_point, 
                self.kmeans_.cluster_centers_, 
                self.coarse_metric
            )[0]
            
            if cluster_distances.max() > 0:
                cluster_distances_norm = cluster_distances / cluster_distances.max()
            else:
                cluster_distances_norm = cluster_distances
            
            # Fine-grained distances
            all_distances = []
            all_indices = []
            
            for cluster_id, cluster_info in self.cluster_labels_.items():
                if len(cluster_info['indices']) == 0:
                    continue
                
                local_distances = self._compute_distance(
                    test_point,
                    cluster_info['X'],
                    self.fine_metric
                )[0]
                
                if local_distances.max() > 0:
                    local_distances_norm = local_distances / local_distances.max()
                else:
                    local_distances_norm = local_distances
                
                cluster_weight_value = cluster_distances_norm[cluster_id]
                combined_distances = (
                    self.cluster_weight * cluster_weight_value +
                    (1 - self.cluster_weight) * local_distances_norm
                )
                
                all_distances.extend(combined_distances)
                all_indices.extend(cluster_info['indices'])
            
            all_distances = np.array(all_distances)
            all_indices = np.array(all_indices)
            
            k = min(self.n_neighbors, len(all_distances))
            nearest_indices = np.argpartition(all_distances, k-1)[:k]
            
            neighbor_labels = self.y_train_[all_indices[nearest_indices]]
            neighbor_distances = all_distances[nearest_indices]
            
            # Inverse distance weighting
            weights = 1.0 / (neighbor_distances + 1e-10)
            
            # Calculate weighted probabilities
            class_weights = {cls: 0.0 for cls in self.classes_}
            for label, weight in zip(neighbor_labels, weights):
                class_weights[label] += weight
            
            total_weight = sum(class_weights.values())
            class_probs = np.array([class_weights[cls] / total_weight for cls in self.classes_])
            
            probabilities.append(class_probs)
        
        return np.array(probabilities)