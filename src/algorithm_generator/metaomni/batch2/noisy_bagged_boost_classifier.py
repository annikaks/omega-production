import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from joblib import Parallel, delayed


class NoisyBaggedBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that combines bagging's parallel training with boosting-style
    sequential noise injection. Each tree is trained on a bootstrap sample with
    controlled noise that decreases across the ensemble.
    
    Parameters
    ----------
    n_estimators : int, default=50
        The number of trees in the ensemble.
    
    base_estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then DecisionTreeClassifier is used.
    
    initial_noise : float, default=0.3
        Initial noise level for the first estimator (as fraction of feature std).
    
    noise_decay : float, default=0.9
        Multiplicative decay factor for noise across estimators.
    
    max_samples : float or int, default=1.0
        The number of samples to draw for each base estimator.
        If float, then draw max_samples * n_samples samples.
    
    max_features : float or int, default=1.0
        The number of features to consider for each base estimator.
        If float, then draw max_features * n_features features.
    
    bootstrap : bool, default=True
        Whether samples are drawn with replacement.
    
    n_jobs : int, default=None
        The number of jobs to run in parallel for fit.
    
    random_state : int, default=None
        Controls the random seed.
    """
    
    def __init__(
        self,
        n_estimators=50,
        base_estimator=None,
        initial_noise=0.3,
        noise_decay=0.9,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        n_jobs=None,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.initial_noise = initial_noise
        self.noise_decay = noise_decay
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    def _get_noise_level(self, estimator_idx):
        """Calculate noise level for a given estimator index."""
        return self.initial_noise * (self.noise_decay ** estimator_idx)
    
    def _inject_noise(self, X, noise_level, feature_std, rng):
        """Inject Gaussian noise into features."""
        noise = rng.normal(0, 1, X.shape)
        # Scale noise by feature standard deviation and noise level
        scaled_noise = noise * feature_std * noise_level
        return X + scaled_noise
    
    def _make_bootstrap_sample(self, X, y, rng):
        """Create a bootstrap sample of the data."""
        n_samples = X.shape[0]
        
        # Determine number of samples
        if isinstance(self.max_samples, float):
            n_samples_bootstrap = int(self.max_samples * n_samples)
        else:
            n_samples_bootstrap = min(self.max_samples, n_samples)
        
        # Sample indices
        if self.bootstrap:
            indices = rng.choice(n_samples, n_samples_bootstrap, replace=True)
        else:
            indices = rng.choice(n_samples, n_samples_bootstrap, replace=False)
        
        return X[indices], y[indices], indices
    
    def _select_features(self, n_features, rng):
        """Select a random subset of features."""
        if isinstance(self.max_features, float):
            n_features_subset = max(1, int(self.max_features * n_features))
        else:
            n_features_subset = min(self.max_features, n_features)
        
        if n_features_subset < n_features:
            return rng.choice(n_features, n_features_subset, replace=False)
        else:
            return np.arange(n_features)
    
    def _fit_single_estimator(self, estimator_idx, X, y, feature_std, seed):
        """Fit a single estimator with noise injection."""
        rng = np.random.RandomState(seed)
        
        # Create bootstrap sample
        X_sample, y_sample, _ = self._make_bootstrap_sample(X, y, rng)
        
        # Select features
        feature_indices = self._select_features(X.shape[1], rng)
        X_sample_features = X_sample[:, feature_indices]
        
        # Calculate noise level for this estimator
        noise_level = self._get_noise_level(estimator_idx)
        
        # Inject noise
        X_noisy = self._inject_noise(
            X_sample_features, 
            noise_level, 
            feature_std[feature_indices],
            rng
        )
        
        # Clone and fit estimator
        estimator = clone(self.base_estimator)
        estimator.fit(X_noisy, y_sample)
        
        return estimator, feature_indices
    
    def fit(self, X_train, y_train):
        """
        Fit the ensemble of noisy bagged estimators.
        
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
        
        # Store classes
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_train.shape[1]
        
        # Set base estimator
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier(max_depth=5)
        
        # Calculate feature standard deviations for noise scaling
        self.feature_std_ = np.std(X_train, axis=0)
        self.feature_std_[self.feature_std_ == 0] = 1.0  # Avoid division by zero
        
        # Generate random seeds for each estimator
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 2**31 - 1, size=self.n_estimators)
        
        # Fit estimators in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_estimator)(
                i, X_train, y_train, self.feature_std_, seeds[i]
            )
            for i in range(self.n_estimators)
        )
        
        # Store estimators and their feature indices
        self.estimators_ = [est for est, _ in results]
        self.feature_indices_ = [feat_idx for _, feat_idx in results]
        
        return self
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X_test has {X_test.shape[1]} features, "
                f"but model was trained with {self.n_features_in_} features."
            )
        
        # Aggregate predictions from all estimators
        all_proba = np.zeros((X_test.shape[0], self.n_classes_))
        
        for estimator, feature_indices in zip(self.estimators_, self.feature_indices_):
            X_test_features = X_test[:, feature_indices]
            
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X_test_features)
            else:
                # For estimators without predict_proba, use one-hot encoding
                predictions = estimator.predict(X_test_features)
                proba = np.zeros((X_test.shape[0], self.n_classes_))
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
            Test data.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]