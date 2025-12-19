import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


class HierarchicalLDA(BaseEstimator, ClassifierMixin):
    """
    Hierarchical Linear Discriminant Analysis with multiple resolution levels.
    
    This classifier builds a hierarchy of LDA models where coarse-level discriminants
    guide fine-grained classification decisions. Classes are hierarchically clustered,
    and LDA models are trained at each level of the hierarchy.
    
    Parameters
    ----------
    n_levels : int, default=3
        Number of hierarchical levels to use.
    
    solver : str, default='svd'
        Solver to use for LDA at each level.
    
    shrinkage : str or float, default=None
        Shrinkage parameter for LDA.
    
    clustering_method : str, default='ward'
        Method for hierarchical clustering of classes.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    
    hierarchy_ : dict
        Dictionary containing the hierarchical structure and LDA models.
    """
    
    def __init__(self, n_levels=3, solver='svd', shrinkage=None, 
                 clustering_method='ward'):
        self.n_levels = n_levels
        self.solver = solver
        self.shrinkage = shrinkage
        self.clustering_method = clustering_method
    
    def _build_class_hierarchy(self, X, y):
        """Build hierarchical clustering of classes based on their centroids."""
        classes = np.unique(y)
        n_classes = len(classes)
        
        # Compute class centroids
        centroids = np.array([X[y == c].mean(axis=0) for c in classes])
        
        # Perform hierarchical clustering on centroids
        if n_classes > 1:
            distances = pdist(centroids)
            linkage_matrix = linkage(distances, method=self.clustering_method)
            
            # Create hierarchy at different levels
            hierarchy = {}
            max_clusters = min(n_classes, max(2, n_classes // (2 ** (self.n_levels - 1))))
            
            for level in range(self.n_levels):
                if level == self.n_levels - 1:
                    # Finest level: each class is its own cluster
                    n_clusters = n_classes
                else:
                    # Coarser levels: progressively fewer clusters
                    n_clusters = max(2, max_clusters // (2 ** level))
                    n_clusters = min(n_clusters, n_classes)
                
                if n_clusters >= n_classes:
                    cluster_labels = np.arange(n_classes)
                else:
                    cluster_labels = fcluster(linkage_matrix, n_clusters, 
                                             criterion='maxclust') - 1
                
                hierarchy[level] = {
                    'n_clusters': n_clusters,
                    'cluster_labels': cluster_labels,
                    'class_to_cluster': dict(zip(classes, cluster_labels))
                }
            
            return hierarchy
        else:
            # Single class case
            hierarchy = {}
            for level in range(self.n_levels):
                hierarchy[level] = {
                    'n_clusters': 1,
                    'cluster_labels': np.array([0]),
                    'class_to_cluster': {classes[0]: 0}
                }
            return hierarchy
    
    def _train_level_lda(self, X, y, level_info):
        """Train LDA model for a specific hierarchy level."""
        # Map classes to clusters for this level
        y_clustered = np.array([level_info['class_to_cluster'][label] 
                               for label in y])
        
        # Train LDA if there are multiple clusters
        if level_info['n_clusters'] > 1:
            lda = LinearDiscriminantAnalysis(
                solver=self.solver,
                shrinkage=self.shrinkage if self.solver == 'lsqr' else None
            )
            lda.fit(X, y_clustered)
            return lda
        else:
            return None
    
    def _train_cluster_specific_models(self, X, y, level, parent_cluster=None):
        """Train LDA models specific to each cluster at a given level."""
        models = {}
        
        if level >= self.n_levels - 1:
            # At finest level, train final classifiers for each cluster
            level_info = self.hierarchy_[level]
            
            for cluster_id in range(level_info['n_clusters']):
                # Get classes belonging to this cluster
                cluster_classes = [c for c, cl in level_info['class_to_cluster'].items() 
                                  if cl == cluster_id]
                
                if len(cluster_classes) > 1:
                    # Get samples for these classes
                    mask = np.isin(y, cluster_classes)
                    if np.sum(mask) > 0:
                        X_cluster = X[mask]
                        y_cluster = y[mask]
                        
                        lda = LinearDiscriminantAnalysis(
                            solver=self.solver,
                            shrinkage=self.shrinkage if self.solver == 'lsqr' else None
                        )
                        lda.fit(X_cluster, y_cluster)
                        models[cluster_id] = lda
                elif len(cluster_classes) == 1:
                    # Single class in cluster
                    models[cluster_id] = cluster_classes[0]
        
        return models
    
    def fit(self, X, y):
        """
        Fit the Hierarchical LDA model.
        
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
        # Validate input
        X, y = check_X_y(X, y)
        
        # Store classes
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Build class hierarchy
        self.hierarchy_ = self._build_class_hierarchy(X, y)
        
        # Train LDA models at each level
        self.level_models_ = {}
        for level in range(self.n_levels):
            level_info = self.hierarchy_[level]
            lda_model = self._train_level_lda(X, y, level_info)
            self.level_models_[level] = lda_model
        
        # Train cluster-specific models at finest level
        self.fine_models_ = self._train_cluster_specific_models(
            X, y, self.n_levels - 1
        )
        
        return self
    
    def _predict_hierarchical(self, X):
        """Predict using hierarchical approach."""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=self.classes_.dtype)
        
        # Start with all samples
        active_samples = np.ones(n_samples, dtype=bool)
        cluster_assignments = np.zeros(n_samples, dtype=int)
        
        # Navigate through hierarchy levels
        for level in range(self.n_levels - 1):
            lda_model = self.level_models_[level]
            
            if lda_model is not None and np.any(active_samples):
                # Predict cluster at this level
                cluster_pred = lda_model.predict(X[active_samples])
                cluster_assignments[active_samples] = cluster_pred
        
        # Final prediction using fine-grained models
        for cluster_id, model in self.fine_models_.items():
            mask = cluster_assignments == cluster_id
            
            if np.any(mask):
                if isinstance(model, LinearDiscriminantAnalysis):
                    predictions[mask] = model.predict(X[mask])
                else:
                    # Single class in cluster
                    predictions[mask] = model
        
        # Handle any remaining samples (fallback)
        if np.any(predictions == 0) and len(self.classes_) > 0:
            # Use the most common class as fallback
            level_info = self.hierarchy_[self.n_levels - 1]
            for i in range(n_samples):
                if predictions[i] == 0 or predictions[i] not in self.classes_:
                    # Assign to first class in the predicted cluster
                    cluster = cluster_assignments[i]
                    cluster_classes = [c for c, cl in level_info['class_to_cluster'].items() 
                                      if cl == cluster]
                    if cluster_classes:
                        predictions[i] = cluster_classes[0]
                    else:
                        predictions[i] = self.classes_[0]
        
        return predictions
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fit has been called
        check_is_fitted(self, ['hierarchy_', 'level_models_', 'fine_models_'])
        
        # Validate input
        X = check_array(X)
        
        # Perform hierarchical prediction
        return self._predict_hierarchical(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ['hierarchy_', 'level_models_', 'fine_models_'])
        X = check_array(X)
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        
        # Get cluster assignments
        active_samples = np.ones(n_samples, dtype=bool)
        cluster_assignments = np.zeros(n_samples, dtype=int)
        cluster_proba = np.ones(n_samples)
        
        # Navigate through hierarchy
        for level in range(self.n_levels - 1):
            lda_model = self.level_models_[level]
            
            if lda_model is not None and np.any(active_samples):
                cluster_pred = lda_model.predict(X[active_samples])
                cluster_prob = lda_model.predict_proba(X[active_samples])
                
                cluster_assignments[active_samples] = cluster_pred
                cluster_proba[active_samples] *= cluster_prob.max(axis=1)
        
        # Get final probabilities
        for cluster_id, model in self.fine_models_.items():
            mask = cluster_assignments == cluster_id
            
            if np.any(mask):
                if isinstance(model, LinearDiscriminantAnalysis):
                    fine_proba = model.predict_proba(X[mask])
                    
                    # Map to global class indices
                    for i, cls in enumerate(model.classes_):
                        cls_idx = np.where(self.classes_ == cls)[0][0]
                        proba[mask, cls_idx] = (fine_proba[:, i] * 
                                               cluster_proba[mask])
                else:
                    # Single class
                    cls_idx = np.where(self.classes_ == model)[0][0]
                    proba[mask, cls_idx] = cluster_proba[mask]
        
        # Normalize probabilities
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        proba /= row_sums
        
        return proba