import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from scipy.special import logsumexp


class CorrelatedFeatureNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Naive Bayes classifier with learned pairwise feature correlations.
    
    This classifier extends the standard Naive Bayes by learning pairwise
    correlation weights between features and decomposing variance into
    correlated and independent components.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Smoothing parameter for variance estimation
    correlation_strength : float, default=0.5
        Weight for correlation component vs independent component (0 to 1)
    max_correlations : int, default=10
        Maximum number of top correlations to consider per feature
    """
    
    def __init__(self, alpha=1.0, correlation_strength=0.5, max_correlations=10):
        self.alpha = alpha
        self.correlation_strength = correlation_strength
        self.max_correlations = max_correlations
        
    def fit(self, X_train, y_train):
        """
        Fit the classifier by learning class priors, feature statistics,
        and pairwise correlation weights.
        
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
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train)
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        # Compute class priors
        self.class_counts_ = np.bincount(y_encoded, minlength=n_classes)
        self.class_priors_ = np.log(self.class_counts_ / len(y))
        
        # Initialize storage for per-class statistics
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        self.correlation_weights_ = []
        self.independent_variances_ = np.zeros((n_classes, n_features))
        self.correlated_variances_ = np.zeros((n_classes, n_features))
        
        # Compute statistics for each class
        for c in range(n_classes):
            X_c = X[y_encoded == c]
            
            # Compute means and variances
            self.means_[c] = np.mean(X_c, axis=0)
            self.variances_[c] = np.var(X_c, axis=0) + self.alpha
            
            # Compute pairwise correlations
            if len(X_c) > 1:
                # Center the data
                X_centered = X_c - self.means_[c]
                
                # Compute correlation matrix
                cov_matrix = np.cov(X_centered.T)
                std_devs = np.sqrt(np.diag(cov_matrix))
                
                # Avoid division by zero
                std_devs = np.where(std_devs > 1e-10, std_devs, 1.0)
                
                # Correlation matrix
                corr_matrix = cov_matrix / (std_devs[:, None] * std_devs[None, :])
                np.fill_diagonal(corr_matrix, 0)  # Remove self-correlations
                
                # Store top correlations for each feature
                class_corr_weights = {}
                for i in range(n_features):
                    # Get absolute correlations for feature i
                    abs_corrs = np.abs(corr_matrix[i])
                    
                    # Get top k correlated features
                    top_indices = np.argsort(abs_corrs)[-self.max_correlations:]
                    top_indices = top_indices[abs_corrs[top_indices] > 0.01]  # Filter weak correlations
                    
                    if len(top_indices) > 0:
                        weights = abs_corrs[top_indices]
                        class_corr_weights[i] = (top_indices, weights)
                
                self.correlation_weights_.append(class_corr_weights)
                
                # Decompose variance into correlated and independent components
                for i in range(n_features):
                    if i in class_corr_weights:
                        indices, weights = class_corr_weights[i]
                        # Correlated variance: weighted by correlation strength
                        corr_var = np.sum(weights) * self.variances_[c, i] * self.correlation_strength
                        self.correlated_variances_[c, i] = corr_var
                        self.independent_variances_[c, i] = self.variances_[c, i] - corr_var
                    else:
                        self.independent_variances_[c, i] = self.variances_[c, i]
                        self.correlated_variances_[c, i] = 0.0
                    
                    # Ensure positive variances
                    self.independent_variances_[c, i] = max(self.independent_variances_[c, i], self.alpha)
            else:
                self.correlation_weights_.append({})
                self.independent_variances_[c] = self.variances_[c]
                self.correlated_variances_[c] = 0.0
        
        return self
    
    def _compute_log_likelihood(self, X, class_idx):
        """
        Compute log likelihood for a given class with correlation adjustments.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
        class_idx : int
            Class index
            
        Returns
        -------
        log_likelihood : array of shape (n_samples,)
            Log likelihood for each sample
        """
        n_samples, n_features = X.shape
        
        # Compute differences from mean
        diff = X - self.means_[class_idx]
        
        # Independent component (standard Gaussian log likelihood)
        independent_log_prob = -0.5 * np.sum(
            np.log(2 * np.pi * self.independent_variances_[class_idx]) +
            (diff ** 2) / self.independent_variances_[class_idx],
            axis=1
        )
        
        # Correlated component adjustment
        correlated_adjustment = np.zeros(n_samples)
        
        if class_idx < len(self.correlation_weights_):
            corr_weights = self.correlation_weights_[class_idx]
            
            for i, (indices, weights) in corr_weights.items():
                if len(indices) > 0:
                    # Compute correlation-based adjustment
                    # Features that are correlated should have similar deviations
                    feature_dev = diff[:, i]
                    correlated_devs = diff[:, indices]
                    
                    # Weighted similarity measure
                    similarity = np.sum(
                        weights * np.exp(-0.5 * (feature_dev[:, None] - correlated_devs) ** 2 / 
                                        (self.correlated_variances_[class_idx, i] + self.alpha)),
                        axis=1
                    )
                    
                    # Add log of similarity as adjustment
                    correlated_adjustment += np.log(similarity + 1e-10) * self.correlation_strength
        
        return independent_log_prob + correlated_adjustment
    
    def predict_proba(self, X_test):
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
        X = np.asarray(X_test, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Compute log probabilities for each class
        log_probs = np.zeros((n_samples, n_classes))
        
        for c in range(n_classes):
            log_likelihood = self._compute_log_likelihood(X, c)
            log_probs[:, c] = self.class_priors_[c] + log_likelihood
        
        # Normalize to get probabilities
        log_probs_normalized = log_probs - logsumexp(log_probs, axis=1, keepdims=True)
        
        return np.exp(log_probs_normalized)
    
    def predict(self, X_test):
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
        y_pred_encoded = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(y_pred_encoded)
    
    def predict_log_proba(self, X_test):
        """
        Predict log class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        log_proba : array of shape (n_samples, n_classes)
            Log class probabilities
        """
        return np.log(self.predict_proba(X_test) + 1e-10)