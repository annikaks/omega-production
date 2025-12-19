import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.special import expit, softmax


class HierarchicalBayesianClassifier(BaseEstimator, ClassifierMixin):
    """
    Hierarchical Bayesian Classifier with feature grouping.
    
    Features are organized into groups where related features share
    hierarchical prior distributions. This allows information sharing
    within feature groups while maintaining flexibility.
    Supports both binary and multi-class classification.
    
    Parameters
    ----------
    feature_groups : list of lists, default=None
        List where each element is a list of feature indices belonging to the same group.
        If None, each feature is its own group.
    
    prior_variance : float, default=1.0
        Variance of the hyperprior on group means.
    
    group_variance : float, default=1.0
        Variance of the prior within each group.
    
    max_iter : int, default=100
        Maximum number of iterations for optimization.
    
    tol : float, default=1e-4
        Tolerance for convergence.
    
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    
    regularization : float, default=0.1
        L2 regularization strength.
    
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, feature_groups=None, prior_variance=1.0, 
                 group_variance=1.0, max_iter=100, tol=1e-4,
                 learning_rate=0.01, regularization=0.1, random_state=None):
        self.feature_groups = feature_groups
        self.prior_variance = prior_variance
        self.group_variance = group_variance
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.random_state = random_state
    
    def _initialize_groups(self, n_features):
        """Initialize feature groups if not provided."""
        if self.feature_groups is None:
            # Each feature is its own group
            self.feature_groups_ = [[i] for i in range(n_features)]
        else:
            self.feature_groups_ = self.feature_groups
            # Validate groups
            all_features = set()
            for group in self.feature_groups_:
                all_features.update(group)
            if len(all_features) != n_features:
                raise ValueError("Feature groups must cover all features exactly once")
        
        self.n_groups_ = len(self.feature_groups_)
        
        # Create mapping from feature index to group index
        self.feature_to_group_ = {}
        for group_idx, group in enumerate(self.feature_groups_):
            for feature_idx in group:
                self.feature_to_group_[feature_idx] = group_idx
    
    def _initialize_parameters(self, n_features, n_classes):
        """Initialize model parameters."""
        rng = np.random.RandomState(self.random_state)
        
        # Group-level means (hyperparameters) - one set per class
        self.group_means_ = rng.normal(0, 0.1, size=(n_classes, self.n_groups_))
        
        # Feature-level weights - one set per class
        self.coef_ = np.zeros((n_classes, n_features))
        for class_idx in range(n_classes):
            for group_idx, group in enumerate(self.feature_groups_):
                for feature_idx in group:
                    self.coef_[class_idx, feature_idx] = rng.normal(
                        self.group_means_[class_idx, group_idx], 0.1
                    )
        
        # Intercept - one per class
        self.intercept_ = np.zeros(n_classes)
    
    def _compute_hierarchical_prior_penalty(self):
        """Compute penalty from hierarchical prior."""
        penalty = 0.0
        
        # Penalty on group means (hyperprior)
        penalty += np.sum(self.group_means_**2) / (2 * self.prior_variance)
        
        # Penalty on feature weights relative to group means
        for class_idx in range(self.n_classes_):
            for group_idx, group in enumerate(self.feature_groups_):
                group_mean = self.group_means_[class_idx, group_idx]
                for feature_idx in group:
                    deviation = self.coef_[class_idx, feature_idx] - group_mean
                    penalty += deviation**2 / (2 * self.group_variance)
        
        return penalty
    
    def _compute_hierarchical_prior_gradient(self):
        """Compute gradient of hierarchical prior penalty."""
        grad_coef = np.zeros_like(self.coef_)
        grad_group_means = np.zeros_like(self.group_means_)
        
        # Gradient w.r.t. group means
        for class_idx in range(self.n_classes_):
            for group_idx, group in enumerate(self.feature_groups_):
                # Hyperprior contribution
                grad_group_means[class_idx, group_idx] = (
                    self.group_means_[class_idx, group_idx] / self.prior_variance
                )
                
                # Group prior contribution (from features)
                for feature_idx in group:
                    deviation = (self.coef_[class_idx, feature_idx] - 
                               self.group_means_[class_idx, group_idx])
                    grad_group_means[class_idx, group_idx] -= deviation / self.group_variance
        
        # Gradient w.r.t. feature weights
        for class_idx in range(self.n_classes_):
            for group_idx, group in enumerate(self.feature_groups_):
                for feature_idx in group:
                    deviation = (self.coef_[class_idx, feature_idx] - 
                               self.group_means_[class_idx, group_idx])
                    grad_coef[class_idx, feature_idx] = deviation / self.group_variance
        
        return grad_coef, grad_group_means
    
    def _compute_loss(self, X, y_onehot):
        """Compute total loss including likelihood and hierarchical prior."""
        # Compute logits
        logits = X @ self.coef_.T + self.intercept_
        
        # Softmax cross-entropy loss
        log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
        log_likelihood = np.mean(np.sum(y_onehot * log_probs, axis=1))
        
        # Hierarchical prior penalty
        prior_penalty = self._compute_hierarchical_prior_penalty()
        
        # Total loss (negative log posterior)
        loss = -log_likelihood + self.regularization * prior_penalty
        
        return loss
    
    def fit(self, X, y):
        """
        Fit the hierarchical Bayesian classifier.
        
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
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        # Create label mapping
        self.label_to_idx_ = {label: idx for idx, label in enumerate(self.classes_)}
        
        # Convert labels to one-hot encoding
        y_indices = np.array([self.label_to_idx_[label] for label in y])
        y_onehot = np.zeros((len(y), self.n_classes_))
        y_onehot[np.arange(len(y)), y_indices] = 1
        
        n_samples, n_features = X.shape
        
        # Initialize feature groups
        self._initialize_groups(n_features)
        
        # Initialize parameters
        self._initialize_parameters(n_features, self.n_classes_)
        
        # Optimization loop
        prev_loss = float('inf')
        
        for iteration in range(self.max_iter):
            # Compute logits and probabilities
            logits = X @ self.coef_.T + self.intercept_
            probs = softmax(logits, axis=1)
            
            # Compute gradients for cross-entropy loss
            grad_factor = probs - y_onehot  # (n_samples, n_classes)
            grad_coef_likelihood = grad_factor.T @ X / n_samples  # (n_classes, n_features)
            grad_intercept_likelihood = np.mean(grad_factor, axis=0)  # (n_classes,)
            
            # Compute hierarchical prior gradients
            grad_coef_prior, grad_group_means_prior = self._compute_hierarchical_prior_gradient()
            
            # Total gradients
            grad_coef = grad_coef_likelihood + self.regularization * grad_coef_prior
            grad_intercept = grad_intercept_likelihood
            grad_group_means = self.regularization * grad_group_means_prior
            
            # Update parameters with gradient descent
            self.coef_ -= self.learning_rate * grad_coef
            self.intercept_ -= self.learning_rate * grad_intercept
            self.group_means_ -= self.learning_rate * grad_group_means
            
            # Check convergence
            current_loss = self._compute_loss(X, y_onehot)
            
            if abs(prev_loss - current_loss) < self.tol:
                break
            
            prev_loss = current_loss
        
        self.n_features_in_ = n_features
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
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
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        
        # Compute logits
        logits = X @ self.coef_.T + self.intercept_
        
        # Convert to probabilities using softmax
        probs = softmax(logits, axis=1)
        
        return probs
    
    def predict(self, X):
        """
        Predict class labels.
        
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
        predicted_indices = np.argmax(proba, axis=1)
        return self.classes_[predicted_indices]
    
    def decision_function(self, X):
        """
        Compute decision function.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        decision : array of shape (n_samples, n_classes)
            Decision function values.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        return X @ self.coef_.T + self.intercept_
    
    def get_group_statistics(self):
        """
        Get statistics about feature groups.
        
        Returns
        -------
        stats : dict
            Dictionary containing group means and feature weights per group.
        """
        check_is_fitted(self)
        
        stats = {
            'group_means': self.group_means_,
            'groups': {}
        }
        
        for group_idx, group in enumerate(self.feature_groups_):
            stats['groups'][group_idx] = {
                'features': group,
                'means_per_class': self.group_means_[:, group_idx],
                'weights_per_class': {}
            }
            
            for class_idx in range(self.n_classes_):
                group_weights = [self.coef_[class_idx, i] for i in group]
                stats['groups'][group_idx]['weights_per_class'][class_idx] = {
                    'weights': group_weights,
                    'mean': np.mean(group_weights),
                    'std': np.std(group_weights)
                }
        
        return stats