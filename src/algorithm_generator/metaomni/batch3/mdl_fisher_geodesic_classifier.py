import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.linalg import eigh
from scipy.spatial.distance import mahalanobis
from itertools import combinations


class MDLFisherGeodesicClassifier(BaseEstimator, ClassifierMixin):
    """
    Minimum Description Length Fisher Geodesic Classifier.
    
    Uses MDL principle to prune Fisher metric components by selecting only
    geodesic paths that contribute to optimal class separation compression ratio.
    
    Parameters
    ----------
    n_components : int, default=None
        Number of geodesic components to retain. If None, determined by MDL.
    mdl_penalty : float, default=1.0
        Penalty weight for model complexity in MDL criterion.
    regularization : float, default=1e-6
        Regularization parameter for covariance matrix inversion.
    max_components : int, default=None
        Maximum number of components to consider. If None, uses min(n_features, n_samples).
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    geodesic_components_ : ndarray
        Selected geodesic components based on MDL.
    fisher_metric_ : ndarray
        Fisher information metric matrix.
    class_means_ : ndarray
        Mean vectors for each class in geodesic space.
    """
    
    def __init__(self, n_components=None, mdl_penalty=1.0, 
                 regularization=1e-6, max_components=None):
        self.n_components = n_components
        self.mdl_penalty = mdl_penalty
        self.regularization = regularization
        self.max_components = max_components
    
    def _compute_fisher_metric(self, X, y):
        """Compute Fisher information metric from class statistics."""
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]
        
        # Compute class-wise statistics
        class_means = []
        class_covs = []
        class_priors = []
        
        for c in classes:
            X_c = X[y == c]
            class_means.append(np.mean(X_c, axis=0))
            cov = np.cov(X_c.T) + self.regularization * np.eye(n_features)
            class_covs.append(cov)
            class_priors.append(len(X_c) / len(X))
        
        # Compute pooled covariance (within-class scatter)
        pooled_cov = np.zeros((n_features, n_features))
        for i, c in enumerate(classes):
            pooled_cov += class_priors[i] * class_covs[i]
        
        # Compute between-class scatter
        overall_mean = np.mean(X, axis=0)
        between_scatter = np.zeros((n_features, n_features))
        for i, c in enumerate(classes):
            mean_diff = (class_means[i] - overall_mean).reshape(-1, 1)
            between_scatter += class_priors[i] * (mean_diff @ mean_diff.T)
        
        # Fisher metric combines both scatters
        fisher_metric = np.linalg.inv(pooled_cov + self.regularization * np.eye(n_features)) @ between_scatter
        
        return fisher_metric, class_means, pooled_cov, class_priors
    
    def _compute_geodesic_paths(self, fisher_metric):
        """Extract geodesic paths via eigendecomposition of Fisher metric."""
        eigenvalues, eigenvectors = eigh(fisher_metric)
        
        # Sort by eigenvalue magnitude (descending)
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def _compute_mdl_score(self, X_projected, y, n_components, n_features):
        """
        Compute MDL score: data encoding cost + model complexity cost.
        Lower is better.
        """
        n_samples = len(X_projected)
        classes = np.unique(y)
        n_classes = len(classes)
        
        # Data encoding cost: negative log-likelihood (simplified)
        data_cost = 0.0
        for c in classes:
            X_c = X_projected[y == c]
            if len(X_c) > 1:
                cov = np.cov(X_c.T) + self.regularization * np.eye(n_components)
                # Gaussian encoding cost
                sign, logdet = np.linalg.slogdet(cov)
                if sign > 0:
                    data_cost += 0.5 * len(X_c) * (n_components * np.log(2 * np.pi) + logdet)
                    for x in X_c:
                        mean = np.mean(X_c, axis=0)
                        diff = x - mean
                        try:
                            data_cost += 0.5 * diff @ np.linalg.inv(cov) @ diff
                        except:
                            data_cost += 0.5 * np.sum(diff ** 2)
        
        # Model complexity cost: number of parameters
        # Parameters: n_components projection vectors + class means in projected space
        n_params = n_components * n_features + n_classes * n_components
        model_cost = self.mdl_penalty * n_params * np.log(n_samples)
        
        # Separation quality bonus (negative cost for better separation)
        separation_bonus = 0.0
        class_means_proj = []
        for c in classes:
            X_c = X_projected[y == c]
            if len(X_c) > 0:
                class_means_proj.append(np.mean(X_c, axis=0))
        
        # Compute inter-class distances
        for i, j in combinations(range(len(class_means_proj)), 2):
            dist = np.linalg.norm(class_means_proj[i] - class_means_proj[j])
            separation_bonus -= dist
        
        mdl_score = data_cost + model_cost + separation_bonus
        return mdl_score
    
    def _select_components_by_mdl(self, X, y, eigenvalues, eigenvectors):
        """Select optimal number of components using MDL principle."""
        n_features = X.shape[1]
        max_comp = self.max_components if self.max_components else min(n_features, len(eigenvalues))
        max_comp = min(max_comp, len(eigenvalues))
        
        if self.n_components is not None:
            return min(self.n_components, max_comp)
        
        best_score = np.inf
        best_n = 1
        
        for n in range(1, max_comp + 1):
            # Project data onto first n components
            projection = eigenvectors[:, :n]
            X_projected = X @ projection
            
            # Compute MDL score
            mdl_score = self._compute_mdl_score(X_projected, y, n, n_features)
            
            if mdl_score < best_score:
                best_score = mdl_score
                best_n = n
        
        return best_n
    
    def fit(self, X_train, y_train):
        """
        Fit the MDL Fisher Geodesic classifier.
        
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
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y_train)
        
        # Store classes
        self.classes_ = unique_labels(y_train)
        
        # Compute Fisher information metric
        fisher_metric, class_means, pooled_cov, class_priors = self._compute_fisher_metric(
            X_train, y_encoded
        )
        self.fisher_metric_ = fisher_metric
        self.pooled_cov_ = pooled_cov
        self.class_priors_ = class_priors
        
        # Extract geodesic paths
        eigenvalues, eigenvectors = self._compute_geodesic_paths(fisher_metric)
        
        # Select components using MDL
        n_selected = self._select_components_by_mdl(
            X_train, y_encoded, eigenvalues, eigenvectors
        )
        
        # Store selected geodesic components
        self.geodesic_components_ = eigenvectors[:, :n_selected]
        self.eigenvalues_ = eigenvalues[:n_selected]
        
        # Project class means to geodesic space
        self.class_means_ = [mean @ self.geodesic_components_ for mean in class_means]
        
        # Project training data and compute class covariances in geodesic space
        X_projected = X_train @ self.geodesic_components_
        self.class_covs_projected_ = []
        for c in np.unique(y_encoded):
            X_c = X_projected[y_encoded == c]
            cov = np.cov(X_c.T) + self.regularization * np.eye(n_selected)
            self.class_covs_projected_.append(cov)
        
        self.n_features_in_ = X_train.shape[1]
        
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
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fitted
        check_is_fitted(self, ['geodesic_components_', 'class_means_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Project test data to geodesic space
        X_projected = X_test @ self.geodesic_components_
        
        # Classify using minimum geodesic distance (Mahalanobis-like)
        predictions = []
        for x in X_projected:
            min_dist = np.inf
            best_class = 0
            
            for i, (mean, cov) in enumerate(zip(self.class_means_, self.class_covs_projected_)):
                try:
                    # Mahalanobis distance in geodesic space
                    diff = x - mean
                    dist = diff @ np.linalg.inv(cov) @ diff
                    # Add prior probability
                    dist -= 2 * np.log(self.class_priors_[i] + 1e-10)
                except:
                    # Fallback to Euclidean distance
                    dist = np.linalg.norm(x - mean)
                
                if dist < min_dist:
                    min_dist = dist
                    best_class = i
            
            predictions.append(best_class)
        
        # Decode labels
        predictions = self.label_encoder_.inverse_transform(predictions)
        
        return predictions