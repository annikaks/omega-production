import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array, check_X_y, check_random_state

class AdaptiveComplexityBagger(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, max_samples=1.0, max_features=1.0, 
                 base_estimator=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.base_estimator = base_estimator
        self.random_state = random_state
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        
        # Check random state
        self.random_state_ = check_random_state(self.random_state)
        
        # Initialize estimators
        self.estimators_ = []
        
        n_samples, n_features = X.shape
        
        # Determine the sample size
        if isinstance(self.max_samples, float):
            max_samples = int(self.max_samples * n_samples)
        else:
            max_samples = self.max_samples
            
        # Determine the feature size
        if isinstance(self.max_features, float):
            max_features = int(self.max_features * n_features)
        else:
            max_features = self.max_features
        
        # Create and fit base estimators
        for _ in range(self.n_estimators):
            # Bootstrap sample
            sample_indices = self.random_state_.randint(0, n_samples, max_samples)
            feature_indices = self.random_state_.choice(n_features, max_features, replace=False)
            
            X_sample = X[sample_indices][:, feature_indices]
            y_sample = y[sample_indices]
            
            # Create estimator
            if self.base_estimator is None:
                estimator = DecisionTreeClassifier(random_state=self.random_state_.randint(0, np.iinfo(np.int32).max))
            else:
                estimator = clone(self.base_estimator)
            
            # Fit estimator
            estimator.fit(X_sample, y_sample)
            
            # Store estimator and feature indices
            self.estimators_.append((estimator, feature_indices))
        
        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, ['estimators_', 'classes_'])
        
        # Input validation
        X = check_array(X)
        
        # Collect predictions from all estimators
        predictions = np.zeros((X.shape[0], len(self.estimators_)))
        
        for i, (estimator, feature_indices) in enumerate(self.estimators_):
            predictions[:, i] = estimator.predict(X[:, feature_indices])
        
        # Return the class with the most votes
        return self.classes_.take(np.argmax(
            np.apply_along_axis(np.bincount, 1, predictions.astype(int),
                                None, np.max(predictions.astype(int)) + 1),
            axis=1
        ))