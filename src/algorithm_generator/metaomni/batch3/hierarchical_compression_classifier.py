import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from typing import List, Optional


class HierarchicalCompressionClassifier(BaseEstimator, ClassifierMixin):
    """
    Hierarchical compression classifier that uses coarse-grained models as priors
    for fine-grained descriptions, enabling multi-scale pattern discovery.
    
    The classifier works by:
    1. Creating multiple scales of data representation (coarse to fine)
    2. Learning patterns at each scale
    3. Using coarse-scale predictions as priors for fine-scale models
    4. Combining multi-scale information for final predictions
    
    Parameters
    ----------
    n_scales : int, default=3
        Number of hierarchical scales to use
    compression_ratios : list of float, optional
        Compression ratios for each scale (from coarse to fine)
        If None, uses exponentially decreasing ratios
    n_clusters_per_scale : list of int, optional
        Number of clusters at each scale for pattern discovery
    base_estimator : estimator object, default=LogisticRegression
        Base classifier to use at each scale
    random_state : int, optional
        Random state for reproducibility
    """
    
    def __init__(
        self,
        n_scales: int = 3,
        compression_ratios: Optional[List[float]] = None,
        n_clusters_per_scale: Optional[List[int]] = None,
        base_estimator=None,
        random_state: Optional[int] = None
    ):
        self.n_scales = n_scales
        self.compression_ratios = compression_ratios
        self.n_clusters_per_scale = n_clusters_per_scale
        self.base_estimator = base_estimator
        self.random_state = random_state
        
    def _initialize_scales(self, n_features: int):
        """Initialize compression ratios and cluster numbers for each scale."""
        if self.compression_ratios is None:
            # Exponentially decreasing compression from coarse to fine
            self.compression_ratios_ = [
                max(2, n_features // (2 ** (self.n_scales - i)))
                for i in range(self.n_scales)
            ]
        else:
            self.compression_ratios_ = self.compression_ratios
            
        if self.n_clusters_per_scale is None:
            # Increasing number of clusters from coarse to fine
            self.n_clusters_per_scale_ = [
                min(10 * (i + 1), 100) for i in range(self.n_scales)
            ]
        else:
            self.n_clusters_per_scale_ = self.n_clusters_per_scale
    
    def _create_scale_representation(
        self, X: np.ndarray, scale_idx: int, pca=None
    ) -> tuple:
        """Create compressed representation at a specific scale."""
        n_components = min(
            self.compression_ratios_[scale_idx], 
            X.shape[1],
            X.shape[0]  # Can't have more components than samples
        )
        n_components = max(1, n_components)  # At least 1 component
        
        if pca is None:
            # Fit PCA for this scale
            pca = PCA(n_components=n_components, random_state=self.random_state)
            X_compressed = pca.fit_transform(X)
        else:
            # Transform using fitted PCA
            X_compressed = pca.transform(X)
            
        return X_compressed, pca
    
    def _discover_patterns(
        self, X: np.ndarray, scale_idx: int, kmeans=None
    ) -> tuple:
        """Discover patterns (clusters) at a specific scale."""
        n_clusters = min(
            self.n_clusters_per_scale_[scale_idx], 
            len(X),
            max(2, len(X) // 2)  # Reasonable number of clusters
        )
        n_clusters = max(2, n_clusters)  # At least 2 clusters
        
        if kmeans is None:
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            cluster_labels = kmeans.fit_predict(X)
        else:
            cluster_labels = kmeans.predict(X)
            n_clusters = kmeans.n_clusters
            
        # Create one-hot encoding of cluster assignments
        cluster_features = np.zeros((len(X), n_clusters))
        cluster_features[np.arange(len(X)), cluster_labels] = 1
        
        return cluster_features, kmeans
    
    def _augment_with_prior(
        self, X: np.ndarray, prior_predictions: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Augment features with prior predictions from coarser scale."""
        if prior_predictions is None:
            return X
        
        # Concatenate original features with prior predictions
        return np.hstack([X, prior_predictions])
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the hierarchical compression classifier.
        
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
        self.classes_ = unique_labels(y_train)
        self.n_features_in_ = X_train.shape[1]
        
        # Initialize scales
        self._initialize_scales(self.n_features_in_)
        
        # Initialize storage for scale-specific models
        self.scalers_ = []
        self.scale_compressors_ = []
        self.pattern_discoverers_ = []
        self.scale_classifiers_ = []
        
        # Train models at each scale, from coarse to fine
        prior_predictions = None
        
        for scale_idx in range(self.n_scales):
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            self.scalers_.append(scaler)
            
            # Create compressed representation at this scale
            X_compressed, pca = self._create_scale_representation(
                X_scaled, scale_idx, pca=None
            )
            self.scale_compressors_.append(pca)
            
            # Discover patterns at this scale
            pattern_features, kmeans = self._discover_patterns(
                X_compressed, scale_idx, kmeans=None
            )
            self.pattern_discoverers_.append(kmeans)
            
            # Augment with prior predictions from coarser scale
            X_augmented = self._augment_with_prior(
                np.hstack([X_compressed, pattern_features]),
                prior_predictions
            )
            
            # Train classifier at this scale
            if self.base_estimator is None:
                classifier = LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state
                )
            else:
                # Clone the base estimator
                from sklearn.base import clone
                classifier = clone(self.base_estimator)
            
            classifier.fit(X_augmented, y_train)
            self.scale_classifiers_.append(classifier)
            
            # Get predictions to use as prior for next scale
            prior_proba = classifier.predict_proba(X_augmented)
            prior_predictions = prior_proba
        
        return self
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X_test has {X_test.shape[1]} features, "
                f"but classifier was fitted with {self.n_features_in_} features"
            )
        
        # Propagate through scales
        prior_predictions = None
        scale_probas = []
        
        for scale_idx in range(self.n_scales):
            # Standardize
            X_scaled = self.scalers_[scale_idx].transform(X_test)
            
            # Compress
            X_compressed, _ = self._create_scale_representation(
                X_scaled, scale_idx, pca=self.scale_compressors_[scale_idx]
            )
            
            # Discover patterns
            pattern_features, _ = self._discover_patterns(
                X_compressed, scale_idx, kmeans=self.pattern_discoverers_[scale_idx]
            )
            
            # Augment with prior
            X_augmented = self._augment_with_prior(
                np.hstack([X_compressed, pattern_features]),
                prior_predictions
            )
            
            # Predict
            proba = self.scale_classifiers_[scale_idx].predict_proba(X_augmented)
            scale_probas.append(proba)
            
            # Update prior for next scale
            prior_predictions = proba
        
        # Combine predictions from all scales with increasing weights for finer scales
        weights = np.array([2 ** i for i in range(self.n_scales)])
        weights = weights / weights.sum()
        
        final_proba = np.zeros_like(scale_probas[0])
        for i, proba in enumerate(scale_probas):
            final_proba += weights[i] * proba
            
        return final_proba
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]