import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import gaussian_kde


class KDELeafDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Decision Tree Classifier with Kernel Density Estimation at leaf nodes.
    
    Instead of using majority voting at leaf nodes, this classifier uses KDE
    to capture local probability distributions for more nuanced predictions.
    
    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the decision tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    bandwidth : float or str, default='scott'
        Bandwidth for KDE. Can be a float or 'scott'/'silverman' for automatic selection.
    kde_method : str, default='sklearn'
        Method for KDE: 'sklearn' or 'scipy'.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1,
                 bandwidth='scott', kde_method='sklearn', random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bandwidth = bandwidth
        self.kde_method = kde_method
        self.random_state = random_state
    
    def fit(self, X_train, y_train):
        """
        Fit the KDE-based decision tree classifier.
        
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
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_train.shape[1]
        
        # Build the decision tree structure
        self.tree_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.tree_.fit(X_train, y_train)
        
        # Get leaf node indices for each training sample
        leaf_indices = self.tree_.apply(X_train)
        unique_leaves = np.unique(leaf_indices)
        
        # Store KDE models for each leaf and class combination
        self.leaf_kdes_ = {}
        self.leaf_class_counts_ = {}
        
        for leaf_id in unique_leaves:
            # Get samples in this leaf
            leaf_mask = leaf_indices == leaf_id
            X_leaf = X_train[leaf_mask]
            y_leaf = y_train[leaf_mask]
            
            # Store class counts for this leaf
            class_counts = {}
            for cls in self.classes_:
                class_counts[cls] = np.sum(y_leaf == cls)
            self.leaf_class_counts_[leaf_id] = class_counts
            
            # Build KDE for each class in this leaf
            self.leaf_kdes_[leaf_id] = {}
            
            for cls in self.classes_:
                class_mask = y_leaf == cls
                X_class = X_leaf[class_mask]
                
                if len(X_class) > 0:
                    # Build KDE model for this class
                    if self.kde_method == 'sklearn':
                        kde = self._build_sklearn_kde(X_class)
                    else:
                        kde = self._build_scipy_kde(X_class)
                    
                    self.leaf_kdes_[leaf_id][cls] = kde
                else:
                    self.leaf_kdes_[leaf_id][cls] = None
        
        return self
    
    def _build_sklearn_kde(self, X):
        """Build KDE using sklearn's KernelDensity."""
        if len(X) == 1:
            # For single sample, use small bandwidth
            bw = 0.1
        elif isinstance(self.bandwidth, str):
            # Use Scott's or Silverman's rule
            if self.bandwidth == 'scott':
                n, d = X.shape
                bw = n ** (-1. / (d + 4))
            elif self.bandwidth == 'silverman':
                n, d = X.shape
                bw = (n * (d + 2) / 4.) ** (-1. / (d + 4))
            else:
                bw = 1.0
        else:
            bw = self.bandwidth
        
        kde = KernelDensity(bandwidth=bw, kernel='gaussian')
        kde.fit(X)
        return kde
    
    def _build_scipy_kde(self, X):
        """Build KDE using scipy's gaussian_kde."""
        if len(X) == 1:
            # For single sample, return a simple wrapper
            return {'type': 'single', 'value': X[0]}
        
        # Transpose for scipy (expects features x samples)
        kde = gaussian_kde(X.T, bw_method=self.bandwidth if isinstance(self.bandwidth, str) else None)
        return {'type': 'scipy', 'kde': kde}
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities using KDE at leaf nodes.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        # Get leaf indices for test samples
        leaf_indices = self.tree_.apply(X_test)
        
        # Initialize probability array
        n_samples = X_test.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))
        
        for i, (x, leaf_id) in enumerate(zip(X_test, leaf_indices)):
            x = x.reshape(1, -1)
            
            # Get KDE models for this leaf
            if leaf_id not in self.leaf_kdes_:
                # Fallback to uniform distribution
                proba[i, :] = 1.0 / self.n_classes_
                continue
            
            leaf_kdes = self.leaf_kdes_[leaf_id]
            class_counts = self.leaf_class_counts_[leaf_id]
            
            # Compute likelihood for each class using KDE
            likelihoods = []
            priors = []
            
            for cls_idx, cls in enumerate(self.classes_):
                kde = leaf_kdes.get(cls)
                count = class_counts.get(cls, 0)
                
                if kde is None or count == 0:
                    likelihoods.append(0.0)
                    priors.append(0.0)
                else:
                    # Compute log-likelihood
                    if self.kde_method == 'sklearn':
                        log_likelihood = kde.score_samples(x)[0]
                        likelihood = np.exp(log_likelihood)
                    else:
                        if kde['type'] == 'single':
                            # Simple distance-based likelihood for single sample
                            dist = np.linalg.norm(x - kde['value'])
                            likelihood = np.exp(-dist ** 2)
                        else:
                            likelihood = kde['kde'](x.T)[0]
                    
                    likelihoods.append(likelihood)
                    priors.append(count)
            
            likelihoods = np.array(likelihoods)
            priors = np.array(priors)
            
            # Normalize priors
            if priors.sum() > 0:
                priors = priors / priors.sum()
            else:
                priors = np.ones(self.n_classes_) / self.n_classes_
            
            # Compute posterior probabilities (Bayes' rule)
            posterior = likelihoods * priors
            
            if posterior.sum() > 0:
                posterior = posterior / posterior.sum()
            else:
                # Fallback to prior if all likelihoods are zero
                posterior = priors
            
            proba[i, :] = posterior
        
        return proba
    
    def predict(self, X_test):
        """
        Predict class labels using KDE at leaf nodes.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]