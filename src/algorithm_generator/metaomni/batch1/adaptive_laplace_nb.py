import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class AdaptiveLaplaceNB(BaseEstimator, ClassifierMixin):
    """
    Naive Bayes classifier with adaptive Laplace smoothing.
    
    The smoothing parameter varies per feature based on its empirical stability
    across cross-validation folds. Features with higher variance in their
    probability estimates receive stronger smoothing.
    
    Parameters
    ----------
    n_folds : int, default=5
        Number of cross-validation folds for estimating feature stability.
    
    base_alpha : float, default=1.0
        Base smoothing parameter that gets scaled per feature.
    
    alpha_range : tuple, default=(0.1, 10.0)
        Range for adaptive smoothing parameters (min, max).
    
    random_state : int, default=None
        Random state for cross-validation splits.
    """
    
    def __init__(self, n_folds=5, base_alpha=1.0, alpha_range=(0.1, 10.0), 
                 random_state=None):
        self.n_folds = n_folds
        self.base_alpha = base_alpha
        self.alpha_range = alpha_range
        self.random_state = random_state
    
    def _compute_feature_stability(self, X, y):
        """
        Compute stability scores for each feature across CV folds.
        Lower stability (higher variance) leads to higher smoothing.
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True, 
                   random_state=self.random_state)
        
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        
        # Store probability estimates for each feature across folds
        # Shape: (n_folds, n_classes, n_features)
        fold_probs = np.zeros((self.n_folds, n_classes, n_features))
        
        for fold_idx, (train_idx, _) in enumerate(kf.split(X)):
            X_fold = X[train_idx]
            y_fold = y[train_idx]
            
            for class_idx, class_label in enumerate(self.classes_):
                class_mask = (y_fold == class_label)
                X_class = X_fold[class_mask]
                
                if len(X_class) > 0:
                    # Compute probability of feature=1 for this class
                    prob_1 = np.mean(X_class, axis=0)
                    fold_probs[fold_idx, class_idx, :] = prob_1
                else:
                    # If no samples in this class, use 0.5 as neutral probability
                    fold_probs[fold_idx, class_idx, :] = 0.5
        
        # Compute variance across folds for each feature and class
        # Then average across classes to get per-feature variance
        # Shape after var: (n_classes, n_features)
        variance_per_class_feature = np.var(fold_probs, axis=0)
        # Average across classes: (n_features,)
        variance_per_feature = np.mean(variance_per_class_feature, axis=0)
        
        # Normalize variance to [0, 1]
        max_var = np.max(variance_per_feature)
        if max_var > 1e-10:
            normalized_variance = variance_per_feature / max_var
        else:
            normalized_variance = np.zeros_like(variance_per_feature)
        
        # Map variance to alpha range (higher variance -> higher alpha)
        alpha_min, alpha_max = self.alpha_range
        adaptive_alphas = alpha_min + normalized_variance * (alpha_max - alpha_min)
        
        return adaptive_alphas
    
    def fit(self, X, y):
        """
        Fit the Adaptive Laplace Naive Bayes classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (binary or continuous features).
        
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        # Compute adaptive smoothing parameters
        self.feature_alphas_ = self._compute_feature_stability(X, y)
        
        # Compute class priors
        self.class_prior_ = np.zeros(self.n_classes_)
        for idx, class_label in enumerate(self.classes_):
            self.class_prior_[idx] = np.mean(y == class_label)
        
        # Compute feature probabilities with adaptive smoothing
        self.feature_log_prob_ = np.zeros((self.n_classes_, self.n_features_, 2))
        
        for idx, class_label in enumerate(self.classes_):
            class_mask = (y == class_label)
            X_class = X[class_mask]
            n_class_samples = np.sum(class_mask)
            
            for feature_idx in range(self.n_features_):
                alpha = self.feature_alphas_[feature_idx]
                
                # Count occurrences (assuming binary or will be binarized)
                feature_values = X_class[:, feature_idx]
                
                # For binary features
                count_1 = np.sum(feature_values > 0.5)
                count_0 = n_class_samples - count_1
                
                # Apply adaptive Laplace smoothing
                prob_1 = (count_1 + alpha) / (n_class_samples + 2 * alpha)
                prob_0 = (count_0 + alpha) / (n_class_samples + 2 * alpha)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                self.feature_log_prob_[idx, feature_idx, 1] = np.log(prob_1 + epsilon)
                self.feature_log_prob_[idx, feature_idx, 0] = np.log(prob_0 + epsilon)
        
        self.class_log_prior_ = np.log(self.class_prior_ + 1e-10)
        
        return self
    
    def predict_log_proba(self, X):
        """
        Compute log probabilities for each class.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        log_proba : array-like of shape (n_samples, n_classes)
            Log probabilities for each class.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        n_samples = X.shape[0]
        log_proba = np.zeros((n_samples, self.n_classes_))
        
        for sample_idx in range(n_samples):
            for class_idx in range(self.n_classes_):
                log_prob = self.class_log_prior_[class_idx]
                
                for feature_idx in range(self.n_features_):
                    feature_value = X[sample_idx, feature_idx]
                    
                    # Binarize feature (threshold at 0.5 for continuous)
                    if feature_value > 0.5:
                        log_prob += self.feature_log_prob_[class_idx, feature_idx, 1]
                    else:
                        log_prob += self.feature_log_prob_[class_idx, feature_idx, 0]
                
                log_proba[sample_idx, class_idx] = log_prob
        
        return log_proba
    
    def predict_proba(self, X):
        """
        Compute probabilities for each class.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Probabilities for each class.
        """
        log_proba = self.predict_log_proba(X)
        
        # Normalize log probabilities using log-sum-exp trick
        log_proba_max = np.max(log_proba, axis=1, keepdims=True)
        exp_proba = np.exp(log_proba - log_proba_max)
        proba = exp_proba / np.sum(exp_proba, axis=1, keepdims=True)
        
        return proba
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        log_proba = self.predict_log_proba(X)
        class_indices = np.argmax(log_proba, axis=1)
        
        return self.classes_[class_indices]