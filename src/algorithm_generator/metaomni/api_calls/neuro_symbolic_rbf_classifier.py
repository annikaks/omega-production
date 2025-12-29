import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from collections import defaultdict


class NeuroSymbolicRBFClassifier(BaseEstimator):
    def __init__(self, n_rbf_centers=10, rbf_gamma=1.0, n_gmm_components=5, 
                 gmm_max_iter=100, gmm_tol=1e-3, rule_confidence_threshold=0.6,
                 min_rule_support=5, random_state=None):
        """
        Neuro-Symbolic RBF Classifier combining RBF networks with symbolic rule induction.
        
        Parameters:
        -----------
        n_rbf_centers : int
            Number of RBF centers
        rbf_gamma : float
            RBF kernel width parameter
        n_gmm_components : int
            Number of Gaussian components for density estimation
        gmm_max_iter : int
            Maximum iterations for GMM fitting
        gmm_tol : float
            Convergence tolerance for GMM
        rule_confidence_threshold : float
            Minimum confidence for rule acceptance
        min_rule_support : int
            Minimum support count for rule generation
        random_state : int or None
            Random seed for reproducibility
        """
        self.n_rbf_centers = n_rbf_centers
        self.rbf_gamma = rbf_gamma
        self.n_gmm_components = n_gmm_components
        self.gmm_max_iter = gmm_max_iter
        self.gmm_tol = gmm_tol
        self.rule_confidence_threshold = rule_confidence_threshold
        self.min_rule_support = min_rule_support
        self.random_state = random_state
        
    def fit(self, X_train, y_train):
        """
        Fit the neuro-symbolic classifier.
        
        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target labels
        """
        self.X_train_ = np.array(X_train)
        self.y_train_ = np.array(y_train)
        self.classes_ = np.unique(y_train)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Step 1: Initialize RBF centers using K-means
        self._fit_rbf_network()
        
        # Step 2: Transform training data to RBF feature space
        self.X_rbf_train_ = self._rbf_transform(self.X_train_)
        
        # Step 3: Fit GMM for density estimation in RBF space
        self._fit_gmm(self.X_rbf_train_)
        
        # Step 4: Induce symbolic rules from RBF features
        self._induce_rules()
        
        return self
    
    def _fit_rbf_network(self):
        """Initialize RBF centers using K-means clustering."""
        n_centers = min(self.n_rbf_centers, len(self.X_train_))
        kmeans = KMeans(n_clusters=n_centers, random_state=self.random_state, n_init=10)
        kmeans.fit(self.X_train_)
        self.rbf_centers_ = kmeans.cluster_centers_
        
    def _rbf_transform(self, X):
        """Transform data to RBF feature space using Gaussian kernels."""
        distances = cdist(X, self.rbf_centers_, metric='euclidean')
        rbf_features = np.exp(-self.rbf_gamma * distances ** 2)
        return rbf_features
    
    def _fit_gmm(self, X):
        """Fit custom Gaussian Mixture Model for density estimation."""
        n_samples, n_features = X.shape
        n_components = min(self.n_gmm_components, n_samples)
        
        # Initialize GMM parameters
        kmeans = KMeans(n_clusters=n_components, random_state=self.random_state, n_init=10)
        kmeans.fit(X)
        
        self.gmm_means_ = kmeans.cluster_centers_
        self.gmm_covariances_ = np.array([np.cov(X.T) + 1e-6 * np.eye(n_features) 
                                          for _ in range(n_components)])
        self.gmm_weights_ = np.ones(n_components) / n_components
        
        # EM algorithm for GMM
        for iteration in range(self.gmm_max_iter):
            # E-step: compute responsibilities
            responsibilities = self._compute_responsibilities(X)
            
            # M-step: update parameters
            old_means = self.gmm_means_.copy()
            
            Nk = responsibilities.sum(axis=0) + 1e-10
            self.gmm_weights_ = Nk / n_samples
            
            for k in range(n_components):
                self.gmm_means_[k] = (responsibilities[:, k:k+1] * X).sum(axis=0) / Nk[k]
                diff = X - self.gmm_means_[k]
                self.gmm_covariances_[k] = (responsibilities[:, k:k+1] * diff).T @ diff / Nk[k]
                self.gmm_covariances_[k] += 1e-6 * np.eye(n_features)
            
            # Check convergence
            if np.linalg.norm(self.gmm_means_ - old_means) < self.gmm_tol:
                break
    
    def _compute_responsibilities(self, X):
        """Compute responsibilities for GMM."""
        n_samples = X.shape[0]
        n_components = len(self.gmm_weights_)
        responsibilities = np.zeros((n_samples, n_components))
        
        for k in range(n_components):
            diff = X - self.gmm_means_[k]
            try:
                cov_inv = np.linalg.inv(self.gmm_covariances_[k])
                cov_det = np.linalg.det(self.gmm_covariances_[k])
                
                if cov_det <= 0:
                    cov_det = 1e-10
                
                exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
                normalization = 1.0 / np.sqrt((2 * np.pi) ** X.shape[1] * cov_det)
                responsibilities[:, k] = self.gmm_weights_[k] * normalization * np.exp(exponent)
            except np.linalg.LinAlgError:
                # If covariance is singular, use small uniform responsibility
                responsibilities[:, k] = 1e-10
        
        # Normalize responsibilities
        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True) + 1e-10
        responsibilities /= responsibilities_sum
        
        return responsibilities
    
    def _compute_density(self, X_rbf):
        """Compute local density using fitted GMM."""
        responsibilities = self._compute_responsibilities(X_rbf)
        density = responsibilities.sum(axis=1)
        return density
    
    def _induce_rules(self):
        """Induce symbolic rules from RBF features."""
        self.rules_ = []
        
        # Generate rules based on RBF feature patterns
        for class_label in self.classes_:
            class_mask = self.y_train_ == class_label
            class_samples = self.X_rbf_train_[class_mask]
            
            if len(class_samples) < self.min_rule_support:
                continue
            
            # Find discriminative RBF features for this class
            for rbf_idx in range(self.rbf_centers_.shape[0]):
                feature_values = class_samples[:, rbf_idx]
                
                # Create rules based on high activation
                high_threshold = np.percentile(self.X_rbf_train_[:, rbf_idx], 75)
                
                if np.mean(feature_values > high_threshold) > 0.5:
                    # Count support and confidence
                    rule_mask = self.X_rbf_train_[:, rbf_idx] > high_threshold
                    support = rule_mask.sum()
                    
                    if support >= self.min_rule_support:
                        correct = (self.y_train_[rule_mask] == class_label).sum()
                        confidence = correct / support if support > 0 else 0
                        
                        if confidence >= self.rule_confidence_threshold:
                            rule = {
                                'rbf_idx': rbf_idx,
                                'threshold': high_threshold,
                                'operator': '>',
                                'class': class_label,
                                'confidence': confidence,
                                'support': support
                            }
                            self.rules_.append(rule)
        
        # Sort rules by confidence
        self.rules_.sort(key=lambda x: x['confidence'], reverse=True)
    
    def predict(self, X_test):
        """
        Predict class labels for test data.
        
        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        X_test = np.array(X_test)
        X_rbf_test = self._rbf_transform(X_test)
        
        # Compute local density for each test sample
        densities = self._compute_density(X_rbf_test)
        
        # Normalize densities to use as weights
        density_weights = densities / (densities.max() + 1e-10)
        
        predictions = []
        
        for i, x_rbf in enumerate(X_rbf_test):
            # Apply rules and collect votes
            class_votes = defaultdict(float)
            
            for rule in self.rules_:
                rbf_value = x_rbf[rule['rbf_idx']]
                
                if rule['operator'] == '>' and rbf_value > rule['threshold']:
                    # Weight rule by confidence and local density
                    weight = rule['confidence'] * (0.5 + 0.5 * density_weights[i])
                    class_votes[rule['class']] += weight
                elif rule['operator'] == '<' and rbf_value < rule['threshold']:
                    weight = rule['confidence'] * (0.5 + 0.5 * density_weights[i])
                    class_votes[rule['class']] += weight
            
            # If no rules fired, use nearest neighbor in RBF space
            if not class_votes:
                distances = cdist([x_rbf], self.X_rbf_train_, metric='euclidean')[0]
                nearest_idx = np.argmin(distances)
                predictions.append(self.y_train_[nearest_idx])
            else:
                # Predict class with highest weighted vote
                predicted_class = max(class_votes.items(), key=lambda x: x[1])[0]
                predictions.append(predicted_class)
        
        return np.array(predictions)