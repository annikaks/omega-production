import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

class BVOptimizedBagger(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0,
                 max_features=1.0, bootstrap=True, bootstrap_features=False,
                 n_jobs=None, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        
        if self.base_estimator is None:
            self.base_estimator_ = DecisionTreeClassifier()
        else:
            self.base_estimator_ = self.base_estimator
        
        # Create a list to store the optimized estimators
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
        
        # Generate random seeds for each estimator
        random_seeds = np.random.randint(np.iinfo(np.int32).max, size=self.n_estimators)
        
        for i in range(self.n_estimators):
            estimator = self.base_estimator_.__class__(**self.base_estimator_.get_params())
            
            if self.bootstrap:
                indices = np.random.choice(n_samples, max_samples, replace=True)
            else:
                indices = np.random.choice(n_samples, max_samples, replace=False)
            
            sample_X = X[indices]
            sample_y = y[indices]
            
            if self.bootstrap_features:
                feature_indices = np.random.choice(n_features, max_features, replace=True)
            else:
                feature_indices = np.random.choice(n_features, max_features, replace=False)
            
            sample_X = sample_X[:, feature_indices]
            
            # Optimize the estimator using cross-validation
            best_score = -np.inf
            best_params = {}
            
            for max_depth in range(1, 11):
                estimator.set_params(max_depth=max_depth)
                score = np.mean(cross_val_score(estimator, sample_X, sample_y, cv=5))
                
                if score > best_score:
                    best_score = score
                    best_params = {'max_depth': max_depth}
            
            estimator.set_params(**best_params)
            estimator.fit(sample_X, sample_y)
            
            self.estimators_.append((estimator, feature_indices))
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        predictions = np.zeros((X.shape[0], len(self.classes_)))
        
        for estimator, feature_indices in self.estimators_:
            predictions += estimator.predict_proba(X[:, feature_indices])
        
        predictions /= len(self.estimators_)
        
        return self.classes_[np.argmax(predictions, axis=1)]