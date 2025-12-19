import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.special import softmax


class CorrelationWeightedNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Naive Bayes classifier with learned pairwise feature correlation weights.
    
    This classifier extends the naive Bayes assumption by learning pairwise
    feature correlations and weighting them based on training data confidence.
    The correlation weights decay as training data size increases, gradually
    approaching standard naive Bayes behavior with more data.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Additive (Laplace/Lidstone) smoothing parameter.
    
    correlation_strength : float, default=1.0
        Controls the strength of pairwise correlation effects.
        Higher values give more weight to learned correlations.
    
    confidence_decay : float, default=0.5
        Controls how quickly correlation weights decay with training data size.
        Higher values lead to faster decay (more confidence in independence).
    
    max_pairs : int, default=100
        Maximum number of feature pairs to consider for correlation learning.
        Limits computational complexity for high-dimensional data.
    """
    
    def __init__(self, alpha=1.0, correlation_strength=1.0, 
                 confidence_decay=0.5, max_pairs=100):
        self.alpha = alpha
        self.correlation_strength = correlation_strength
        self.confidence_decay = confidence_decay
        self.max_pairs = max_pairs
    
    def _compute_confidence_weight(self, n_samples):
        """
        Compute confidence-based decay weight for correlations.
        
        As n_samples increases, this weight decreases, reducing the
        influence of learned correlations in favor of independence assumption.
        """
        return np.exp(-self.confidence_decay * np.log(n_samples + 1))
    
    def _select_top_pairs(self, X, y):
        """
        Select top feature pairs based on mutual information with target.
        """
        n_features = X.shape[1]
        
        # Compute correlation strength for each pair
        pair_scores = []
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # Compute conditional mutual information as proxy for importance
                score = self._compute_pair_importance(X[:, i], X[:, j], y)
                pair_scores.append((score, i, j))
        
        # Sort and select top pairs
        pair_scores.sort(reverse=True)
        selected_pairs = [(i, j) for _, i, j in pair_scores[:self.max_pairs]]
        
        return selected_pairs
    
    def _compute_pair_importance(self, xi, xj, y):
        """
        Compute importance score for a feature pair.
        Uses correlation strength weighted by class-conditional variance.
        """
        # Compute correlation
        corr = np.abs(np.corrcoef(xi, xj)[0, 1])
        
        # Weight by class-conditional variance
        importance = 0
        for c in np.unique(y):
            mask = y == c
            if np.sum(mask) > 1:
                var_i = np.var(xi[mask]) + 1e-10
                var_j = np.var(xj[mask]) + 1e-10
                importance += corr * np.sqrt(var_i * var_j)
        
        return importance
    
    def fit(self, X, y):
        """
        Fit the correlation-weighted Naive Bayes classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        n_samples = X.shape[0]
        
        # Compute confidence weight based on training data size
        self.confidence_weight_ = self._compute_confidence_weight(n_samples)
        
        # Compute class priors
        self.class_prior_ = np.zeros(self.n_classes_)
        for idx, c in enumerate(self.classes_):
            self.class_prior_[idx] = np.sum(y == c) / n_samples
        
        # Compute feature statistics per class (assuming Gaussian features)
        self.theta_ = np.zeros((self.n_classes_, self.n_features_))
        self.sigma_ = np.zeros((self.n_classes_, self.n_features_))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[idx, :] = np.mean(X_c, axis=0)
            self.sigma_[idx, :] = np.var(X_c, axis=0) + self.alpha
        
        # Select and learn pairwise correlations
        self.feature_pairs_ = self._select_top_pairs(X, y)
        
        # Learn pairwise correlation weights for each class
        self.pair_weights_ = {}
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            class_pair_weights = {}
            
            for i, j in self.feature_pairs_:
                # Compute correlation coefficient
                if len(X_c) > 1:
                    corr = np.corrcoef(X_c[:, i], X_c[:, j])[0, 1]
                    # Store correlation as weight
                    class_pair_weights[(i, j)] = corr
                else:
                    class_pair_weights[(i, j)] = 0.0
            
            self.pair_weights_[c] = class_pair_weights
        
        return self
    
    def _compute_log_likelihood(self, X, class_idx):
        """
        Compute log likelihood with correlation adjustments.
        """
        c = self.classes_[class_idx]
        
        # Standard Naive Bayes log likelihood (independence assumption)
        log_likelihood = np.log(self.class_prior_[class_idx])
        
        # Add individual feature contributions
        for j in range(self.n_features_):
            mean = self.theta_[class_idx, j]
            var = self.sigma_[class_idx, j]
            log_likelihood += -0.5 * np.log(2 * np.pi * var)
            log_likelihood += -0.5 * ((X[:, j] - mean) ** 2) / var
        
        # Add pairwise correlation adjustments (decayed by confidence)
        if self.feature_pairs_ and c in self.pair_weights_:
            correlation_adjustment = np.zeros(X.shape[0])
            
            for i, j in self.feature_pairs_:
                weight = self.pair_weights_[c].get((i, j), 0.0)
                
                # Compute interaction term
                xi_norm = (X[:, i] - self.theta_[class_idx, i]) / np.sqrt(self.sigma_[class_idx, i])
                xj_norm = (X[:, j] - self.theta_[class_idx, j]) / np.sqrt(self.sigma_[class_idx, j])
                
                # Add weighted correlation term
                correlation_adjustment += weight * xi_norm * xj_norm
            
            # Apply confidence-weighted correlation adjustment
            log_likelihood += (self.correlation_strength * 
                             self.confidence_weight_ * 
                             correlation_adjustment)
        
        return log_likelihood
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")
        
        # Compute log likelihoods for all classes
        log_likelihoods = np.zeros((X.shape[0], self.n_classes_))
        
        for idx in range(self.n_classes_):
            log_likelihoods[:, idx] = self._compute_log_likelihood(X, idx)
        
        # Convert to probabilities using softmax
        return softmax(log_likelihoods, axis=1)
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True labels.
        
        Returns
        -------
        score : float
            Mean accuracy.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))