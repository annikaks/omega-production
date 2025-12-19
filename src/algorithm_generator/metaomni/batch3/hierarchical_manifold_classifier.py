import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
import warnings
warnings.filterwarnings('ignore')


class HierarchicalManifoldClassifier(BaseEstimator, ClassifierMixin):
    """
    Hierarchical Compression Trees with Manifold Embeddings for Classification.
    
    This classifier creates a hierarchical tree structure that compresses discrete
    symbolic representations into continuous manifold embeddings, enabling hybrid
    structure learning for classification tasks.
    
    Parameters
    ----------
    n_levels : int, default=3
        Number of hierarchical levels in the compression tree
    manifold_dim : int, default=10
        Dimensionality of the continuous manifold embedding
    n_clusters_per_level : int, default=5
        Number of clusters at each hierarchical level
    manifold_method : str, default='isomap'
        Method for manifold learning ('isomap' or 'pca')
    distance_metric : str, default='euclidean'
        Distance metric for clustering and classification
    alpha : float, default=0.5
        Weight balance between symbolic and manifold representations (0 to 1)
    
    Attributes
    ----------
    tree_structure_ : dict
        Hierarchical tree structure mapping symbols to embeddings
    manifold_embeddings_ : dict
        Continuous manifold embeddings at each level
    label_encoder_ : LabelEncoder
        Encoder for class labels
    """
    
    def __init__(self, n_levels=3, manifold_dim=10, n_clusters_per_level=5,
                 manifold_method='isomap', distance_metric='euclidean', alpha=0.5):
        self.n_levels = n_levels
        self.manifold_dim = manifold_dim
        self.n_clusters_per_level = n_clusters_per_level
        self.manifold_method = manifold_method
        self.distance_metric = distance_metric
        self.alpha = alpha
        
    def _create_manifold_embedding(self, X):
        """Create continuous manifold embedding from discrete representations."""
        n_samples, n_features = X.shape
        
        # Adjust manifold dimension if needed
        manifold_dim = min(self.manifold_dim, n_features, n_samples - 1)
        
        if self.manifold_method == 'isomap':
            n_neighbors = min(5, n_samples - 1)
            if n_neighbors > 0:
                embedder = Isomap(n_components=manifold_dim, n_neighbors=n_neighbors)
            else:
                embedder = PCA(n_components=manifold_dim)
        else:
            embedder = PCA(n_components=manifold_dim)
        
        try:
            embedding = embedder.fit_transform(X)
        except:
            # Fallback to PCA if Isomap fails
            embedder = PCA(n_components=manifold_dim)
            embedding = embedder.fit_transform(X)
            
        return embedding, embedder
    
    def _hierarchical_clustering(self, X, n_clusters):
        """Perform hierarchical clustering on data."""
        if len(X) <= n_clusters:
            return np.arange(len(X))
        
        # Compute linkage matrix
        linkage_matrix = linkage(X, method='ward')
        
        # Get cluster assignments
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        return clusters - 1  # Convert to 0-indexed
    
    def _build_compression_tree(self, X, y, level=0):
        """Recursively build hierarchical compression tree."""
        if level >= self.n_levels or len(X) <= 1:
            return {
                'level': level,
                'data': X,
                'labels': y,
                'is_leaf': True,
                'cluster_id': None,
                'children': []
            }
        
        # Create manifold embedding for current level
        embedding, embedder = self._create_manifold_embedding(X)
        
        # Perform hierarchical clustering
        n_clusters = min(self.n_clusters_per_level, len(X))
        cluster_assignments = self._hierarchical_clustering(embedding, n_clusters)
        
        # Build node
        node = {
            'level': level,
            'embedder': embedder,
            'embedding': embedding,
            'data': X,
            'labels': y,
            'is_leaf': False,
            'n_clusters': n_clusters,
            'cluster_assignments': cluster_assignments,
            'children': []
        }
        
        # Recursively build children
        for cluster_id in range(n_clusters):
            mask = cluster_assignments == cluster_id
            if np.sum(mask) > 0:
                child_X = X[mask]
                child_y = y[mask]
                child_node = self._build_compression_tree(child_X, child_y, level + 1)
                child_node['cluster_id'] = cluster_id
                child_node['parent_centroid'] = np.mean(embedding[mask], axis=0)
                node['children'].append(child_node)
        
        return node
    
    def _compute_node_statistics(self, node):
        """Compute statistics for each node in the tree."""
        if node['is_leaf']:
            # Leaf node statistics
            unique_labels, counts = np.unique(node['labels'], return_counts=True)
            node['label_distribution'] = dict(zip(unique_labels, counts))
            node['majority_label'] = unique_labels[np.argmax(counts)]
            node['centroid'] = np.mean(node['data'], axis=0)
        else:
            # Internal node statistics
            node['centroid'] = np.mean(node['data'], axis=0)
            unique_labels, counts = np.unique(node['labels'], return_counts=True)
            node['label_distribution'] = dict(zip(unique_labels, counts))
            node['majority_label'] = unique_labels[np.argmax(counts)]
            
            # Compute cluster centroids in embedding space
            node['cluster_centroids'] = []
            for cluster_id in range(node['n_clusters']):
                mask = node['cluster_assignments'] == cluster_id
                if np.sum(mask) > 0:
                    centroid = np.mean(node['embedding'][mask], axis=0)
                    node['cluster_centroids'].append(centroid)
                else:
                    node['cluster_centroids'].append(None)
            
            # Recursively compute for children
            for child in node['children']:
                self._compute_node_statistics(child)
    
    def fit(self, X_train, y_train):
        """
        Fit the hierarchical manifold classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target labels
            
        Returns
        -------
        self : object
            Fitted classifier
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y_train)
        
        # Build hierarchical compression tree
        self.tree_root_ = self._build_compression_tree(X_train, y_encoded)
        
        # Compute statistics for all nodes
        self._compute_node_statistics(self.tree_root_)
        
        # Store training data for reference
        self.X_train_ = X_train
        self.y_train_ = y_encoded
        
        return self
    
    def _traverse_tree(self, x, node):
        """Traverse tree to find best matching leaf node."""
        if node['is_leaf']:
            return node
        
        # Transform to manifold space
        x_reshaped = x.reshape(1, -1)
        x_embedded = node['embedder'].transform(x_reshaped)[0]
        
        # Find closest cluster centroid
        distances = []
        for i, centroid in enumerate(node['cluster_centroids']):
            if centroid is not None:
                dist = np.linalg.norm(x_embedded - centroid)
                distances.append((dist, i))
        
        if not distances:
            return node
        
        distances.sort()
        closest_cluster = distances[0][1]
        
        # Find corresponding child
        for child in node['children']:
            if child['cluster_id'] == closest_cluster:
                return self._traverse_tree(x, child)
        
        # Fallback to current node
        return node
    
    def _predict_single(self, x):
        """Predict label for a single sample."""
        # Traverse tree to find best matching node
        leaf_node = self._traverse_tree(x, self.tree_root_)
        
        # Symbolic prediction (majority label in leaf)
        symbolic_pred = leaf_node['majority_label']
        
        # Manifold prediction (nearest neighbor in original space)
        distances = cdist(x.reshape(1, -1), leaf_node['data'], metric=self.distance_metric)[0]
        nearest_idx = np.argmin(distances)
        manifold_pred = leaf_node['labels'][nearest_idx]
        
        # Hybrid prediction (weighted combination)
        if self.alpha >= 0.5:
            return symbolic_pred
        else:
            return manifold_pred
    
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
        
        predictions = []
        for x in X_test:
            pred = self._predict_single(x)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Decode labels back to original
        return self.label_encoder_.inverse_transform(predictions)
    
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
        n_classes = len(self.label_encoder_.classes_)
        probas = np.zeros((len(X_test), n_classes))
        
        for i, x in enumerate(X_test):
            leaf_node = self._traverse_tree(x, self.tree_root_)
            
            # Get label distribution in leaf
            label_dist = leaf_node['label_distribution']
            total_samples = sum(label_dist.values())
            
            for label, count in label_dist.items():
                probas[i, label] = count / total_samples
        
        return probas
    
    def get_tree_depth(self, node=None):
        """Get the depth of the compression tree."""
        if node is None:
            node = self.tree_root_
        
        if node['is_leaf']:
            return 1
        
        max_child_depth = 0
        for child in node['children']:
            child_depth = self.get_tree_depth(child)
            max_child_depth = max(max_child_depth, child_depth)
        
        return 1 + max_child_depth