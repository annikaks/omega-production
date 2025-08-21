import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_array, check_X_y, check_random_state
from scipy.stats import entropy

class EntropyGuidedBagger(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0,
                 max_features=1.0, bootstrap=True, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape

        if self.base_estimator is None:
            self.base_estimator_ = DecisionTreeClassifier()
        else:
            self.base_estimator_ = self.base_estimator

        self.estimators_ = []
        self.estimators_features_ = []

        random_state = check_random_state(self.random_state)

        for i in range(self.n_estimators):
            estimator = self.base_estimator_.clone()

            if self.bootstrap:
                indices = random_state.randint(0, n_samples, size=int(self.max_samples * n_samples))
            else:
                indices = np.arange(n_samples)

            sample_counts = np.bincount(y[indices])
            sample_entropy = entropy(sample_counts)

            feature_entropies = []
            for feature in range(n_features):
                feature_counts = np.bincount(X[indices, feature].astype(int))
                feature_entropies.append(entropy(feature_counts))

            feature_weights = np.array(feature_entropies) / sample_entropy
            feature_weights /= np.sum(feature_weights)

            n_features_to_draw = int(self.max_features * n_features)
            selected_features = random_state.choice(n_features, size=n_features_to_draw, 
                                                    replace=False, p=feature_weights)

            estimator.fit(X[indices][:, selected_features], y[indices])

            self.estimators_.append(estimator)
            self.estimators_features_.append(selected_features)

        return self

    def predict(self, X):
        X = check_array(X)
        n_samples = X.shape[0]

        predictions = np.zeros((n_samples, len(self.classes_)))

        for estimator, features in zip(self.estimators_, self.estimators_features_):
            predictions += estimator.predict_proba(X[:, features])

        predictions /= len(self.estimators_)

        return self.classes_[np.argmax(predictions, axis=1)]

    def predict_proba(self, X):
        X = check_array(X)
        n_samples = X.shape[0]

        probas = np.zeros((n_samples, len(self.classes_)))

        for estimator, features in zip(self.estimators_, self.estimators_features_):
            probas += estimator.predict_proba(X[:, features])

        probas /= len(self.estimators_)

        return probas