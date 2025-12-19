import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KernelDensity
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler


class DensityAwareBootstrapClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses density-aware bootstrap sampling to oversample from
    low-density regions, improving decision boundaries in sparse areas.
    
    Parameters
    ----------
    base_estimator : estimator object, default=None
        The base classifier to use. If None, uses DecisionTreeClassifier.
    
    n_estimators : int, default=10
        The number of bootstrap samples/estimators to train.
    
    density_bandwidth : float, default='auto'
        Bandwidth for kernel density estimation. If 'auto', uses Scott's rule.
    
    density_power : float, default=1.0
        Power to raise inverse density weights. Higher values increase
        oversampling from low-density regions.
    
    sample_size : float or int, default=1.0
        If float, proportion of dataset size for each bootstrap sample.
        If int, absolute number of samples.
    
    random_state : int, default=None
        Random state for reproducibility.
    
    normalize : bool, default=True
        Whether to normalize features for density estimation.
    """
    
    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        density_bandwidth='auto',
        density_power=1.0,
        sample_size=1.0,
        random_state=None,
        normalize=True
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.density_bandwidth = density_bandwidth
        self.density_power = density_power
        self.sample_size = sample_size
        self.random_state = random_state
        self.normalize = normalize
    
    def _estimate_density(self, X):
        """Estimate density for each sample using kernel density estimation."""
        if self.normalize:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        
        # Determine bandwidth
        if self.density_bandwidth == 'auto':
            bandwidth = X_scaled.shape[1] ** (-1.0 / (X_scaled.shape[1] + 4)) * X_scaled.std()
        else:
            bandwidth = self.density_bandwidth
        
        # Fit KDE
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(X_scaled)
        
        # Get log density and convert to density
        log_density = kde.score_samples(X_scaled)
        density = np.exp(log_density)
        
        return density
    
    def _compute_sampling_weights(self, X):
        """Compute sampling weights inversely proportional to density."""
        density = self._estimate_density(X)
        
        # Avoid division by zero
        density = np.maximum(density, 1e-10)
        
        # Inverse density with power
        weights = (1.0 / density) ** self.density_power
        
        # Normalize weights
        weights = weights / weights.sum()
        
        return weights
    
    def fit(self, X_train, y_train):
        """
        Fit the density-aware bootstrap classifier.
        
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
        
        # Initialize scaler if normalization is enabled
        if self.normalize:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X_train)
        
        # Compute sampling weights based on density
        sampling_weights = self._compute_sampling_weights(X_train)
        
        # Determine sample size
        n_samples = X_train.shape[0]
        if isinstance(self.sample_size, float):
            bootstrap_size = int(self.sample_size * n_samples)
        else:
            bootstrap_size = self.sample_size
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Train ensemble of estimators
        self.estimators_ = []
        
        for i in range(self.n_estimators):
            # Create bootstrap sample with density-aware sampling
            indices = rng.choice(
                n_samples,
                size=bootstrap_size,
                replace=True,
                p=sampling_weights
            )
            
            X_bootstrap = X_train[indices]
            y_bootstrap = y_train[indices]
            
            # Initialize base estimator
            if self.base_estimator is None:
                estimator = DecisionTreeClassifier(random_state=rng.randint(0, 10000))
            else:
                estimator = self.base_estimator.__class__(**self.base_estimator.get_params())
            
            # Fit estimator
            estimator.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(estimator)
        
        return self
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.
        
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
        
        # Aggregate predictions from all estimators
        all_proba = np.zeros((X_test.shape[0], len(self.classes_)))
        
        for estimator in self.estimators_:
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X_test)
            else:
                # For estimators without predict_proba, use one-hot encoding
                predictions = estimator.predict(X_test)
                proba = np.zeros((X_test.shape[0], len(self.classes_)))
                for i, cls in enumerate(self.classes_):
                    proba[:, i] = (predictions == cls).astype(float)
            
            all_proba += proba
        
        # Average probabilities
        all_proba /= self.n_estimators
        
        return all_proba
    
    def predict(self, X_test):
        """
        Predict class labels for X_test.
        
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