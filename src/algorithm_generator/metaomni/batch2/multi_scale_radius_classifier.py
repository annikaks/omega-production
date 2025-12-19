import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import cdist
from collections import Counter


class MultiScaleRadiusClassifier(BaseEstimator, ClassifierMixin):
    """
    Multi-scale radius search classifier using coarse-grain clustering
    followed by fine-grain local radius search for classification.
    
    Parameters
    ----------
    n_clusters : int, default=10
        Number of coarse-grain clusters to create
    coarse_radius : float, default=1.0
        Radius for coarse-grain region identification
    fine_radius : float, default=0.5
        Radius for fine-grain local search within regions
    min_samples : int, default=1
        Minimum number of samples within fine radius for classification
    metric : str, default='euclidean'
        Distance metric to use
    """
    
    def __init__(self, n_clusters=10, coarse_radius=1.0, fine_radius=0.5,
                 min_samples=1, metric='euclidean'):
        self.n_clusters = n_clusters
        self.coarse_radius = coarse_radius
        self.fine_radius = fine_radius
        self.min_samples = min_samples
        self.metric = metric
    
    def fit(self, X_train, y_train):
        """
        Fit the multi-scale radius classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        
        # Store classes
        self.classes_ = unique_labels(y_train)
        
        # Store training data
        self.X_train_ = X_train
        self.y_train_ = y_train
        
        # Coarse-grain clustering
        self.n_clusters_ = min(self.n_clusters, len(X_train))
        self.kmeans_ = KMeans(n_clusters=self.n_clusters_, random_state=42)
        self.cluster_labels_ = self.kmeans_.fit_predict(X_train)
        
        # Build cluster-to-samples mapping
        self.cluster_samples_ = {}
        for cluster_id in range(self.n_clusters_):
            mask = self.cluster_labels_ == cluster_id
            self.cluster_samples_[cluster_id] = {
                'X': X_train[mask],
                'y': y_train[mask],
                'indices': np.where(mask)[0]
            }
        
        # Build NearestNeighbors for each cluster for fine-grain search
        self.cluster_nn_ = {}
        for cluster_id in range(self.n_clusters_):
            if len(self.cluster_samples_[cluster_id]['X']) > 0:
                nn = NearestNeighbors(metric=self.metric)
                nn.fit(self.cluster_samples_[cluster_id]['X'])
                self.cluster_nn_[cluster_id] = nn
        
        # Build global NearestNeighbors as fallback
        self.global_nn_ = NearestNeighbors(metric=self.metric)
        self.global_nn_.fit(X_train)
        
        return self
    
    def _predict_single(self, x):
        """
        Predict class for a single sample using multi-scale approach.
        
        Parameters
        ----------
        x : array-like of shape (n_features,)
            Single sample to predict
            
        Returns
        -------
        prediction : int
            Predicted class label
        """
        x = x.reshape(1, -1)
        
        # Step 1: Coarse-grain - find nearest cluster centers
        cluster_distances = cdist(x, self.kmeans_.cluster_centers_, 
                                  metric=self.metric)[0]
        
        # Identify relevant clusters within coarse radius
        relevant_clusters = np.where(cluster_distances <= self.coarse_radius)[0]
        
        # If no clusters within coarse radius, use nearest cluster
        if len(relevant_clusters) == 0:
            relevant_clusters = [np.argmin(cluster_distances)]
        
        # Step 2: Fine-grain - search within relevant clusters
        candidate_labels = []
        candidate_distances = []
        
        for cluster_id in relevant_clusters:
            if cluster_id not in self.cluster_nn_:
                continue
                
            # Find neighbors within fine radius in this cluster
            nn = self.cluster_nn_[cluster_id]
            cluster_data = self.cluster_samples_[cluster_id]
            
            # Query radius neighbors
            indices = nn.radius_neighbors(x, radius=self.fine_radius, 
                                         return_distance=False)[0]
            
            if len(indices) >= self.min_samples:
                # Get labels and distances for neighbors
                neighbor_labels = cluster_data['y'][indices]
                neighbor_X = cluster_data['X'][indices]
                distances = cdist(x, neighbor_X, metric=self.metric)[0]
                
                candidate_labels.extend(neighbor_labels)
                candidate_distances.extend(distances)
        
        # Step 3: Make prediction based on fine-grain neighbors
        if len(candidate_labels) >= self.min_samples:
            # Weighted voting by inverse distance
            weights = 1.0 / (np.array(candidate_distances) + 1e-10)
            
            # Count weighted votes for each class
            class_votes = {}
            for label, weight in zip(candidate_labels, weights):
                class_votes[label] = class_votes.get(label, 0) + weight
            
            # Return class with highest weighted vote
            prediction = max(class_votes.items(), key=lambda x: x[1])[0]
        else:
            # Fallback: use global k-nearest neighbors
            distances, indices = self.global_nn_.kneighbors(x, n_neighbors=5)
            neighbor_labels = self.y_train_[indices[0]]
            
            # Simple majority voting
            counter = Counter(neighbor_labels)
            prediction = counter.most_common(1)[0][0]
        
        return prediction
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        # Check if fit has been called
        check_is_fitted(self, ['X_train_', 'y_train_', 'kmeans_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Predict for each sample
        predictions = np.array([self._predict_single(x) for x in X_test])
        
        return predictions
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self, ['X_train_', 'y_train_', 'kmeans_'])
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        
        for i, x in enumerate(X_test):
            x = x.reshape(1, -1)
            
            # Find relevant clusters
            cluster_distances = cdist(x, self.kmeans_.cluster_centers_, 
                                     metric=self.metric)[0]
            relevant_clusters = np.where(cluster_distances <= self.coarse_radius)[0]
            
            if len(relevant_clusters) == 0:
                relevant_clusters = [np.argmin(cluster_distances)]
            
            # Collect neighbors from relevant clusters
            all_labels = []
            all_weights = []
            
            for cluster_id in relevant_clusters:
                if cluster_id not in self.cluster_nn_:
                    continue
                
                nn = self.cluster_nn_[cluster_id]
                cluster_data = self.cluster_samples_[cluster_id]
                
                indices = nn.radius_neighbors(x, radius=self.fine_radius, 
                                             return_distance=False)[0]
                
                if len(indices) > 0:
                    neighbor_labels = cluster_data['y'][indices]
                    neighbor_X = cluster_data['X'][indices]
                    distances = cdist(x, neighbor_X, metric=self.metric)[0]
                    weights = 1.0 / (distances + 1e-10)
                    
                    all_labels.extend(neighbor_labels)
                    all_weights.extend(weights)
            
            # Calculate probabilities
            if len(all_labels) > 0:
                for label, weight in zip(all_labels, all_weights):
                    class_idx = np.where(self.classes_ == label)[0][0]
                    proba[i, class_idx] += weight
                
                # Normalize
                proba[i] /= proba[i].sum()
            else:
                # Fallback to uniform distribution
                proba[i] = 1.0 / n_classes
        
        return proba