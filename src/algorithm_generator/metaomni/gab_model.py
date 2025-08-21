import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array, check_X_y, check_random_state

class GranularityAdaptiveBagger(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0,
                 max_features=1.0, bootstrap=True, bootstrap_features=False,
                 granularity_levels=3, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.granularity_levels = granularity_levels
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        if self.base_estimator is None:
            self.base_estimator_ = DecisionTreeClassifier()
        else:
            self.base_estimator_ = self.base_estimator
        
        self.estimators_ = []
        self.granularities_ = []
        
        n_samples, n_features = X.shape
        
        # Calculate max_samples and max_features
        max_samples = self._validate_max_samples(self.max_samples, n_samples)
        max_features = self._validate_max_features(self.max_features, n_features)
        
        random_state = check_random_state(self.random_state)
        
        for i in range(self.n_estimators):
            estimator = self._make_estimator(append=False, random_state=random_state)
            
            # Choose granularity level
            granularity = random_state.randint(1, self.granularity_levels + 1)
            self.granularities_.append(granularity)
            
            # Sample instances and features
            sample_indices = self._sample_indices(n_samples, max_samples, granularity, random_state)
            feature_indices = self._sample_indices(n_features, max_features, granularity, random_state)
            
            X_train = X[np.ix_(sample_indices, feature_indices)]
            y_train = y[sample_indices]
            
            estimator.fit(X_train, y_train)
            self.estimators_.append(estimator)
        
        return self

    def predict(self, X):
        X = check_array(X)
        n_samples = X.shape[0]
        
        predictions = np.zeros((n_samples, self.n_classes_))
        
        for estimator, granularity in zip(self.estimators_, self.granularities_):
            X_subset = self._subset_features(X, granularity)
            proba = estimator.predict_proba(X_subset)
            predictions += proba
        
        return self.classes_.take(np.argmax(predictions, axis=1), axis=0)

    def _validate_max_samples(self, max_samples, n_samples):
        if isinstance(max_samples, int):
            return max(1, min(max_samples, n_samples))
        elif isinstance(max_samples, float):
            return max(1, int(max_samples * n_samples))

    def _validate_max_features(self, max_features, n_features):
        if isinstance(max_features, int):
            return max(1, min(max_features, n_features))
        elif isinstance(max_features, float):
            return max(1, int(max_features * n_features))

    def _make_estimator(self, append=True, random_state=None):
        estimator = clone(self.base_estimator_)
        estimator.set_params(random_state=random_state)
        if append:
            self.estimators_.append(estimator)
        return estimator

    def _sample_indices(self, n_population, max_count, granularity, random_state):
        if self.bootstrap:
            indices = random_state.randint(0, n_population, size=max_count)
        else:
            indices = random_state.permutation(n_population)[:max_count]
        
        # Apply granularity
        return indices[:max(1, int(len(indices) / granularity))]

    def _subset_features(self, X, granularity):
        n_features = X.shape[1]
        subset_size = max(1, int(n_features / granularity))
        return X[:, :subset_size]