import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class HierarchicalFisherPoolingClassifier(BaseEstimator, ClassifierMixin):
    """
    Hierarchical Fisher Information Pooling Classifier.
    
    Implements multi-scale Fisher vector encoding where:
    - Coarse levels capture global compressed structure via larger spatial pooling
    - Fine levels preserve local discriminative geometry via detailed encoding
    
    Parameters
    ----------
    n_components : int, default=16
        Number of Gaussian components for GMM at finest level
    n_scales : int, default=3
        Number of hierarchical scales (coarse to fine)
    scale_factor : float, default=2.0
        Factor by which components decrease at coarser levels
    spatial_bins : int, default=2
        Number of spatial bins per dimension for spatial pyramid
    kernel : str, default='rbf'
        Kernel type for final SVM classifier
    C : float, default=1.0
        Regularization parameter for SVM
    random_state : int, default=None
        Random state for reproducibility
    """
    
    def __init__(self, n_components=16, n_scales=3, scale_factor=2.0,
                 spatial_bins=2, kernel='rbf', C=1.0, random_state=None):
        self.n_components = n_components
        self.n_scales = n_scales
        self.scale_factor = scale_factor
        self.spatial_bins = spatial_bins
        self.kernel = kernel
        self.C = C
        self.random_state = random_state
    
    def _create_spatial_grid(self, n_samples, n_bins):
        """Create spatial grid assignments for samples."""
        # Simulate spatial layout by clustering feature indices
        indices = np.arange(n_samples)
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        
        # Assign samples to spatial bins
        x_coords = indices % grid_size
        y_coords = indices // grid_size
        
        x_bins = np.digitize(x_coords, np.linspace(0, grid_size, n_bins + 1)[1:-1])
        y_bins = np.digitize(y_coords, np.linspace(0, grid_size, n_bins + 1)[1:-1])
        
        spatial_ids = x_bins * n_bins + y_bins
        return spatial_ids
    
    def _compute_fisher_vector(self, X, gmm):
        """
        Compute Fisher vector encoding for data X given GMM.
        
        Returns gradient statistics w.r.t. mean and covariance.
        """
        n_samples, n_features = X.shape
        n_components = gmm.n_components
        
        # Compute posterior probabilities
        posteriors = gmm.predict_proba(X)  # (n_samples, n_components)
        
        # Get GMM parameters
        means = gmm.means_  # (n_components, n_features)
        covariances = gmm.covariances_  # (n_components, n_features) for 'diag'
        weights = gmm.weights_  # (n_components,)
        
        # Initialize Fisher vector components
        fv_mu = np.zeros((n_components, n_features))
        fv_sigma = np.zeros((n_components, n_features))
        
        # Compute gradients
        for k in range(n_components):
            # Difference from mean
            diff = X - means[k]  # (n_samples, n_features)
            
            # Weighted by posterior
            weighted_post = posteriors[:, k:k+1]  # (n_samples, 1)
            
            # Gradient w.r.t. mean
            fv_mu[k] = np.sum(weighted_post * diff / np.sqrt(covariances[k]), axis=0)
            fv_mu[k] /= (n_samples * np.sqrt(weights[k]))
            
            # Gradient w.r.t. covariance
            normalized_diff_sq = (diff ** 2) / covariances[k]
            fv_sigma[k] = np.sum(weighted_post * (normalized_diff_sq - 1), axis=0)
            fv_sigma[k] /= (n_samples * np.sqrt(2 * weights[k]))
        
        # Concatenate and flatten
        fv = np.concatenate([fv_mu.flatten(), fv_sigma.flatten()])
        
        # Power normalization
        fv = np.sign(fv) * np.sqrt(np.abs(fv))
        
        # L2 normalization
        fv = fv / (np.linalg.norm(fv) + 1e-12)
        
        return fv
    
    def _compute_hierarchical_fisher_vectors(self, X):
        """
        Compute Fisher vectors at multiple scales with spatial pooling.
        """
        n_samples = X.shape[0]
        all_fvs = []
        
        # Process each scale from coarse to fine
        for scale_idx in range(self.n_scales):
            # Determine number of components for this scale
            n_comp = max(2, int(self.n_components / (self.scale_factor ** (self.n_scales - scale_idx - 1))))
            
            gmm = self.gmms_[scale_idx]
            
            # Compute spatial bins for this scale
            n_bins = max(1, self.spatial_bins - scale_idx)
            
            if n_bins == 1:
                # Global pooling
                fv = self._compute_fisher_vector(X, gmm)
                all_fvs.append(fv)
            else:
                # Spatial pyramid pooling
                spatial_ids = self._create_spatial_grid(n_samples, n_bins)
                spatial_fvs = []
                
                for bin_id in range(n_bins * n_bins):
                    mask = spatial_ids == bin_id
                    if np.sum(mask) > 0:
                        X_bin = X[mask]
                        fv_bin = self._compute_fisher_vector(X_bin, gmm)
                        spatial_fvs.append(fv_bin)
                    else:
                        # Empty bin - use zero vector
                        fv_dim = 2 * n_comp * X.shape[1]
                        spatial_fvs.append(np.zeros(fv_dim))
                
                # Concatenate spatial bins
                spatial_fv = np.concatenate(spatial_fvs)
                all_fvs.append(spatial_fv)
        
        # Concatenate all scales
        hierarchical_fv = np.concatenate(all_fvs)
        return hierarchical_fv
    
    def fit(self, X_train, y_train):
        """
        Fit the hierarchical Fisher pooling classifier.
        
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
        
        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_train)
        
        # Train GMMs at multiple scales
        self.gmms_ = []
        
        for scale_idx in range(self.n_scales):
            # Determine number of components for this scale
            n_comp = max(2, int(self.n_components / (self.scale_factor ** (self.n_scales - scale_idx - 1))))
            
            # Train GMM
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type='diag',
                random_state=self.random_state,
                max_iter=100,
                n_init=3
            )
            gmm.fit(X_scaled)
            self.gmms_.append(gmm)
        
        # Compute Fisher vectors for all training samples
        X_fisher = []
        for i in range(len(X_train)):
            fv = self._compute_hierarchical_fisher_vectors(X_scaled[i:i+1])
            X_fisher.append(fv)
        
        X_fisher = np.array(X_fisher)
        
        # Train final classifier
        self.classifier_ = SVC(kernel=self.kernel, C=self.C, random_state=self.random_state)
        self.classifier_.fit(X_fisher, y_train)
        
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
        # Validate input
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        # Standardize features
        X_scaled = self.scaler_.transform(X_test)
        
        # Compute Fisher vectors for all test samples
        X_fisher = []
        for i in range(len(X_test)):
            fv = self._compute_hierarchical_fisher_vectors(X_scaled[i:i+1])
            X_fisher.append(fv)
        
        X_fisher = np.array(X_fisher)
        
        # Predict using trained classifier
        y_pred = self.classifier_.predict(X_fisher)
        
        return y_pred
    
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
            Predicted class probabilities
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        # Standardize features
        X_scaled = self.scaler_.transform(X_test)
        
        # Compute Fisher vectors
        X_fisher = []
        for i in range(len(X_test)):
            fv = self._compute_hierarchical_fisher_vectors(X_scaled[i:i+1])
            X_fisher.append(fv)
        
        X_fisher = np.array(X_fisher)
        
        # Get probabilities if available
        if hasattr(self.classifier_, 'predict_proba'):
            return self.classifier_.predict_proba(X_fisher)
        else:
            # Fallback to decision function
            decision = self.classifier_.decision_function(X_fisher)
            if decision.ndim == 1:
                decision = decision.reshape(-1, 1)
            # Simple softmax
            exp_decision = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_decision / np.sum(exp_decision, axis=1, keepdims=True)