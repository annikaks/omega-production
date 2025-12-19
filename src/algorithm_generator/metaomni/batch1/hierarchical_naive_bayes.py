import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import defaultdict


class HierarchicalNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Hierarchical Naive Bayes classifier that groups features into conditionally
    independent clusters based on mutual information.
    
    Parameters
    ----------
    n_clusters : int, default=None
        Number of feature clusters. If None, automatically determined.
    distance_threshold : float, default=0.5
        Distance threshold for hierarchical clustering. Used if n_clusters is None.
    linkage_method : str, default='average'
        Linkage method for hierarchical clustering.
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter.
    """
    
    def __init__(self, n_clusters=None, distance_threshold=0.5, 
                 linkage_method='average', alpha=1.0):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.linkage_method = linkage_method
        self.alpha = alpha
        
    def _compute_mutual_information_matrix(self, X, y):
        """Compute pairwise mutual information between features."""
        n_features = X.shape[1]
        mi_matrix = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Compute mutual information between features i and j
                mi = mutual_info_score(X[:, i], X[:, j])
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
        
        return mi_matrix
    
    def _cluster_features(self, X, y):
        """Cluster features based on mutual information."""
        n_features = X.shape[1]
        
        if n_features == 1:
            return np.array([0])
        
        # Compute mutual information matrix
        mi_matrix = self._compute_mutual_information_matrix(X, y)
        
        # Convert MI to distance (higher MI = lower distance)
        # Add small epsilon to avoid division by zero
        max_mi = np.max(mi_matrix)
        if max_mi > 0:
            distance_matrix = 1 - (mi_matrix / (max_mi + 1e-10))
        else:
            distance_matrix = np.ones_like(mi_matrix)
        
        # Ensure diagonal is zero
        np.fill_diagonal(distance_matrix, 0)
        
        # Convert to condensed distance matrix for linkage
        condensed_dist = squareform(distance_matrix, checks=False)
        
        # Perform hierarchical clustering
        Z = linkage(condensed_dist, method=self.linkage_method)
        
        # Cut the dendrogram to form clusters
        if self.n_clusters is not None:
            labels = fcluster(Z, self.n_clusters, criterion='maxclust')
        else:
            labels = fcluster(Z, self.distance_threshold, criterion='distance')
        
        # Convert to 0-indexed
        return labels - 1
    
    def _discretize_features(self, X):
        """Discretize continuous features into bins."""
        X_discrete = np.zeros_like(X, dtype=int)
        
        for i in range(X.shape[1]):
            feature = X[:, i]
            unique_vals = np.unique(feature)
            
            # If already discrete (few unique values), keep as is
            if len(unique_vals) <= 10:
                # Map to consecutive integers
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                X_discrete[:, i] = np.array([mapping[val] for val in feature])
            else:
                # Bin continuous features
                n_bins = min(10, len(unique_vals))
                X_discrete[:, i] = np.digitize(feature, 
                                               bins=np.percentile(feature, 
                                                                 np.linspace(0, 100, n_bins)))
        
        return X_discrete
    
    def fit(self, X_train, y_train):
        """
        Fit the Hierarchical Naive Bayes classifier.
        
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
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y_train)
        
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X_train.shape[1]
        
        # Discretize features
        X_discrete = self._discretize_features(X_train)
        self.feature_ranges_ = [np.max(X_discrete[:, i]) + 1 
                               for i in range(self.n_features_)]
        
        # Cluster features based on mutual information
        self.feature_clusters_ = self._cluster_features(X_discrete, y_encoded)
        self.n_clusters_ = len(np.unique(self.feature_clusters_))
        
        # Compute class priors
        self.class_prior_ = np.zeros(self.n_classes_)
        for c in range(self.n_classes_):
            self.class_prior_[c] = np.sum(y_encoded == c) / len(y_encoded)
        
        # Compute conditional probabilities for each cluster
        self.cluster_probs_ = []
        
        for cluster_id in range(self.n_clusters_):
            # Get features in this cluster
            cluster_features = np.where(self.feature_clusters_ == cluster_id)[0]
            
            # Store probabilities for this cluster
            cluster_prob_dict = {}
            
            for c in range(self.n_classes_):
                X_class = X_discrete[y_encoded == c][:, cluster_features]
                n_samples_class = X_class.shape[0]
                
                # Compute joint probability for features in cluster
                # For simplicity, we still assume independence within cluster
                feature_probs = []
                
                for feat_idx, feat in enumerate(cluster_features):
                    n_values = self.feature_ranges_[feat]
                    probs = np.zeros(n_values)
                    
                    for val in range(n_values):
                        count = np.sum(X_class[:, feat_idx] == val)
                        # Laplace smoothing
                        probs[val] = (count + self.alpha) / (n_samples_class + self.alpha * n_values)
                    
                    feature_probs.append(probs)
                
                cluster_prob_dict[c] = feature_probs
            
            self.cluster_probs_.append(cluster_prob_dict)
        
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
        X_test = np.asarray(X_test)
        X_discrete = self._discretize_features(X_test)
        n_samples = X_test.shape[0]
        
        log_proba = np.zeros((n_samples, self.n_classes_))
        
        for c in range(self.n_classes_):
            # Start with class prior
            log_proba[:, c] = np.log(self.class_prior_[c] + 1e-10)
            
            # Add log probabilities from each cluster
            for cluster_id in range(self.n_clusters_):
                cluster_features = np.where(self.feature_clusters_ == cluster_id)[0]
                
                for feat_idx, feat in enumerate(cluster_features):
                    feature_probs = self.cluster_probs_[cluster_id][c][feat_idx]
                    
                    for i in range(n_samples):
                        val = X_discrete[i, feat]
                        # Clip value to valid range
                        val = min(val, len(feature_probs) - 1)
                        log_proba[i, c] += np.log(feature_probs[val] + 1e-10)
        
        # Convert log probabilities to probabilities
        # Subtract max for numerical stability
        log_proba -= np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba /= np.sum(proba, axis=1, keepdims=True)
        
        return proba
    
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
        y_pred_encoded = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(y_pred_encoded)