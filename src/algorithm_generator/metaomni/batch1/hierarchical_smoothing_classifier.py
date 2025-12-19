import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from collections import defaultdict, Counter


class HierarchicalSmoothingClassifier(BaseEstimator, ClassifierMixin):
    """
    A hierarchical smoothing classifier that transitions from simple Laplace smoothing
    to context-dependent smoothing based on local data density.
    
    Parameters
    ----------
    alpha : float, default=1.0
        Laplace smoothing parameter for low-density regions
    density_threshold : float, default=10.0
        Minimum number of samples in a region to use context-dependent smoothing
    n_neighbors : int, default=5
        Number of neighbors to consider for density estimation
    context_weight : float, default=0.7
        Weight for context-dependent smoothing (1 - context_weight for Laplace)
    """
    
    def __init__(self, alpha=1.0, density_threshold=10.0, n_neighbors=5, context_weight=0.7):
        self.alpha = alpha
        self.density_threshold = density_threshold
        self.n_neighbors = n_neighbors
        self.context_weight = context_weight
    
    def _compute_density(self, X, point):
        """Compute local density around a point using k-nearest neighbors."""
        distances = np.linalg.norm(X - point, axis=1)
        k = min(self.n_neighbors, len(X))
        knn_distances = np.partition(distances, k-1)[:k]
        density = k / (np.mean(knn_distances) + 1e-10)
        return density
    
    def _laplace_smoothing(self, feature_counts, class_count, n_features):
        """Apply simple Laplace (additive) smoothing."""
        smoothed_probs = {}
        for feature_val, count in feature_counts.items():
            smoothed_probs[feature_val] = (count + self.alpha) / (
                class_count + self.alpha * n_features
            )
        # Default probability for unseen features
        default_prob = self.alpha / (class_count + self.alpha * n_features)
        return smoothed_probs, default_prob
    
    def _context_dependent_smoothing(self, feature_counts, class_count, 
                                     context_features, n_features):
        """Apply context-dependent smoothing based on feature co-occurrence."""
        smoothed_probs = {}
        
        # Build context model: P(feature | other_features, class)
        for feature_val, count in feature_counts.items():
            # Base probability
            base_prob = count / max(class_count, 1)
            
            # Context adjustment based on feature co-occurrence
            context_boost = 1.0
            if context_features:
                # Calculate how often this feature appears with context features
                context_count = sum(1 for cf in context_features 
                                  if cf in self.feature_cooccurrence_.get(feature_val, set()))
                context_boost = 1.0 + (context_count / max(len(context_features), 1))
            
            # Combine base probability with context
            smoothed_probs[feature_val] = base_prob * context_boost
        
        # Normalize
        total = sum(smoothed_probs.values()) + 1e-10
        smoothed_probs = {k: v / total for k, v in smoothed_probs.items()}
        
        default_prob = 1.0 / (n_features * 10)  # Lower default for high-density regions
        return smoothed_probs, default_prob
    
    def fit(self, X, y):
        """
        Fit the hierarchical smoothing classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Store training data for density estimation
        self.X_train_ = X
        self.y_train_ = y
        
        # Compute class priors
        self.class_counts_ = Counter(y)
        self.class_priors_ = {
            cls: count / len(y) for cls, count in self.class_counts_.items()
        }
        
        # Build feature statistics per class
        self.feature_stats_ = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for i, (sample, label) in enumerate(zip(X, y)):
            for feature_idx, feature_val in enumerate(sample):
                # Discretize continuous features for counting
                discretized_val = np.round(feature_val, decimals=2)
                self.feature_stats_[label][feature_idx][discretized_val] += 1
        
        # Build feature co-occurrence matrix for context-dependent smoothing
        self.feature_cooccurrence_ = defaultdict(set)
        for sample in X:
            discretized_sample = [np.round(val, decimals=2) for val in sample]
            for i, val_i in enumerate(discretized_sample):
                for j, val_j in enumerate(discretized_sample):
                    if i != j:
                        self.feature_cooccurrence_[(i, val_i)].add((j, val_j))
        
        # Compute density map for training data
        self.density_map_ = np.array([
            self._compute_density(X, point) for point in X
        ])
        
        return self
    
    def _predict_single(self, x):
        """Predict class for a single sample."""
        # Estimate local density
        density = self._compute_density(self.X_train_, x)
        
        # Determine smoothing strategy based on density
        use_context = density >= self.density_threshold
        
        # Discretize features
        discretized_x = [np.round(val, decimals=2) for val in x]
        context_features = [(i, val) for i, val in enumerate(discretized_x)]
        
        class_scores = {}
        
        for cls in self.classes_:
            # Start with class prior
            log_prob = np.log(self.class_priors_[cls] + 1e-10)
            
            # Add feature likelihoods
            for feature_idx, feature_val in enumerate(discretized_x):
                feature_counts = self.feature_stats_[cls][feature_idx]
                class_count = self.class_counts_[cls]
                n_unique_features = len(set(self.X_train_[:, feature_idx]))
                
                if use_context:
                    # Use context-dependent smoothing
                    smoothed_probs, default_prob = self._context_dependent_smoothing(
                        feature_counts, class_count, context_features, n_unique_features
                    )
                    prob = smoothed_probs.get(feature_val, default_prob)
                    
                    # Blend with Laplace smoothing
                    laplace_probs, laplace_default = self._laplace_smoothing(
                        feature_counts, class_count, n_unique_features
                    )
                    laplace_prob = laplace_probs.get(feature_val, laplace_default)
                    
                    prob = (self.context_weight * prob + 
                           (1 - self.context_weight) * laplace_prob)
                else:
                    # Use simple Laplace smoothing
                    smoothed_probs, default_prob = self._laplace_smoothing(
                        feature_counts, class_count, n_unique_features
                    )
                    prob = smoothed_probs.get(feature_val, default_prob)
                
                log_prob += np.log(prob + 1e-10)
            
            class_scores[cls] = log_prob
        
        # Return class with highest score
        return max(class_scores, key=class_scores.get)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels
        """
        check_is_fitted(self)
        X = check_array(X)
        
        predictions = np.array([self._predict_single(x) for x in X])
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self)
        X = check_array(X)
        
        probas = []
        for x in X:
            density = self._compute_density(self.X_train_, x)
            use_context = density >= self.density_threshold
            
            discretized_x = [np.round(val, decimals=2) for val in x]
            context_features = [(i, val) for i, val in enumerate(discretized_x)]
            
            class_scores = {}
            
            for cls in self.classes_:
                log_prob = np.log(self.class_priors_[cls] + 1e-10)
                
                for feature_idx, feature_val in enumerate(discretized_x):
                    feature_counts = self.feature_stats_[cls][feature_idx]
                    class_count = self.class_counts_[cls]
                    n_unique_features = len(set(self.X_train_[:, feature_idx]))
                    
                    if use_context:
                        smoothed_probs, default_prob = self._context_dependent_smoothing(
                            feature_counts, class_count, context_features, n_unique_features
                        )
                        prob = smoothed_probs.get(feature_val, default_prob)
                        
                        laplace_probs, laplace_default = self._laplace_smoothing(
                            feature_counts, class_count, n_unique_features
                        )
                        laplace_prob = laplace_probs.get(feature_val, laplace_default)
                        
                        prob = (self.context_weight * prob + 
                               (1 - self.context_weight) * laplace_prob)
                    else:
                        smoothed_probs, default_prob = self._laplace_smoothing(
                            feature_counts, class_count, n_unique_features
                        )
                        prob = smoothed_probs.get(feature_val, default_prob)
                    
                    log_prob += np.log(prob + 1e-10)
                
                class_scores[cls] = log_prob
            
            # Convert log probabilities to probabilities
            scores = np.array([class_scores[cls] for cls in self.classes_])
            scores = np.exp(scores - np.max(scores))  # Numerical stability
            scores /= np.sum(scores)
            probas.append(scores)
        
        return np.array(probas)