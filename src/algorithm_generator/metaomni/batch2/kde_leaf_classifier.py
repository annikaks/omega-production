import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class KDELeafClassifier(BaseEstimator, ClassifierMixin):
    """
    A decision tree classifier with KDE at leaf nodes for probabilistic predictions.
    
    This classifier builds a decision tree and fits a kernel density estimator
    for each class at each leaf node, providing smooth probabilistic predictions
    instead of hard class assignments.
    
    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the decision tree.
    min_samples_leaf : int, default=5
        Minimum number of samples required at a leaf node.
    bandwidth : float or str, default='scott'
        Bandwidth for KDE. Can be a float or 'scott'/'silverman' for automatic selection.
    kernel : str, default='gaussian'
        Kernel to use for KDE: 'gaussian', 'tophat', 'epanechnikov', etc.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, max_depth=5, min_samples_leaf=5, bandwidth='scott', 
                 kernel='gaussian', random_state=None):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.random_state = random_state
    
    def _compute_bandwidth(self, X):
        """Compute bandwidth using Scott's or Silverman's rule."""
        n, d = X.shape
        if self.bandwidth == 'scott':
            return n ** (-1. / (d + 4))
        elif self.bandwidth == 'silverman':
            return (n * (d + 2) / 4.) ** (-1. / (d + 4))
        else:
            return self.bandwidth
    
    def _build_leaf_kdes(self, X, y, tree):
        """Build KDE models for each class at each leaf node."""
        leaf_kdes = {}
        
        # Get leaf node indices for all training samples
        leaf_ids = tree.apply(X)
        unique_leaves = np.unique(leaf_ids)
        
        for leaf_id in unique_leaves:
            # Get samples in this leaf
            leaf_mask = leaf_ids == leaf_id
            X_leaf = X[leaf_mask]
            y_leaf = y[leaf_mask]
            
            # Build KDE for each class in this leaf
            leaf_kdes[leaf_id] = {}
            for class_label in self.classes_:
                class_mask = y_leaf == class_label
                X_class = X_leaf[class_mask]
                
                if len(X_class) > 0:
                    # Compute bandwidth for this class subset
                    if isinstance(self.bandwidth, str):
                        bw = self._compute_bandwidth(X_class)
                    else:
                        bw = self.bandwidth
                    
                    # Fit KDE for this class
                    kde = KernelDensity(bandwidth=bw, kernel=self.kernel)
                    kde.fit(X_class)
                    leaf_kdes[leaf_id][class_label] = {
                        'kde': kde,
                        'prior': np.sum(class_mask) / len(y_leaf)
                    }
                else:
                    # No samples of this class in leaf
                    leaf_kdes[leaf_id][class_label] = {
                        'kde': None,
                        'prior': 0.0
                    }
        
        return leaf_kdes
    
    def fit(self, X_train, y_train):
        """
        Fit the KDE leaf classifier.
        
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
        self.classes_ = unique_labels(y_train)
        self.n_features_in_ = X_train.shape[1]
        
        # Build decision tree
        self.tree_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.tree_.fit(X_train, y_train)
        
        # Build KDE models at each leaf
        self.leaf_kdes_ = self._build_leaf_kdes(X_train, y_train, self.tree_)
        
        return self
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities using KDE at leaf nodes.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ['tree_', 'leaf_kdes_'])
        X_test = check_array(X_test)
        
        # Get leaf assignments for test samples
        leaf_ids = self.tree_.apply(X_test)
        n_samples = X_test.shape[0]
        n_classes = len(self.classes_)
        
        proba = np.zeros((n_samples, n_classes))
        
        for i, (x, leaf_id) in enumerate(zip(X_test, leaf_ids)):
            x = x.reshape(1, -1)
            
            if leaf_id not in self.leaf_kdes_:
                # Fallback to uniform distribution
                proba[i, :] = 1.0 / n_classes
                continue
            
            log_densities = []
            priors = []
            
            for j, class_label in enumerate(self.classes_):
                kde_info = self.leaf_kdes_[leaf_id][class_label]
                
                if kde_info['kde'] is not None and kde_info['prior'] > 0:
                    # Compute log density
                    log_dens = kde_info['kde'].score_samples(x)[0]
                    log_densities.append(log_dens)
                    priors.append(kde_info['prior'])
                else:
                    # No samples of this class in leaf
                    log_densities.append(-np.inf)
                    priors.append(1e-10)
            
            log_densities = np.array(log_densities)
            priors = np.array(priors)
            
            # Compute posterior probabilities using Bayes' rule
            # P(class|x) ∝ P(x|class) * P(class)
            log_posteriors = log_densities + np.log(priors)
            
            # Normalize using log-sum-exp trick for numerical stability
            max_log_post = np.max(log_posteriors[np.isfinite(log_posteriors)])
            if np.isfinite(max_log_post):
                exp_log_post = np.exp(log_posteriors - max_log_post)
                proba[i, :] = exp_log_post / np.sum(exp_log_post)
            else:
                # Fallback to priors if all densities are zero
                proba[i, :] = priors / np.sum(priors)
        
        return proba
    
    def predict(self, X_test):
        """
        Predict class labels.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]