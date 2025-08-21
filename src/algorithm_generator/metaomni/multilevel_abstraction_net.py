import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class MultiLevelAbstractionNet(BaseEstimator, ClassifierMixin):
    def __init__(self, n_levels=3, classifiers=None):
        self.n_levels = n_levels
        self.classifiers = classifiers or [
            LogisticRegression(),
            SVC(kernel='rbf', probability=True),
            RandomForestClassifier(),
            MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
        ]
        self.scalers = [StandardScaler() for _ in range(n_levels)]
        self.models = []
        
    def _create_abstraction_features(self, X, level):
        if level == 0:
            return X
        elif level == 1:
            return np.hstack([X, np.mean(X, axis=1).reshape(-1, 1), np.std(X, axis=1).reshape(-1, 1)])
        elif level == 2:
            return np.hstack([X, np.percentile(X, [25, 50, 75], axis=1).T])
        else:
            raise ValueError("Unsupported abstraction level")

    def fit(self, X, y):
        self.models = []
        for level in range(self.n_levels):
            X_level = self._create_abstraction_features(X, level)
            X_scaled = self.scalers[level].fit_transform(X_level)
            
            level_models = []
            for clf in self.classifiers:
                model = clf.fit(X_scaled, y)
                level_models.append(model)
            
            self.models.append(level_models)
        
        return self

    def predict(self, X):
        predictions = []
        for level in range(self.n_levels):
            X_level = self._create_abstraction_features(X, level)
            X_scaled = self.scalers[level].transform(X_level)
            
            level_predictions = []
            for model in self.models[level]:
                level_predictions.append(model.predict_proba(X_scaled))
            
            level_predictions = np.mean(level_predictions, axis=0)
            predictions.append(level_predictions)
        
        final_predictions = np.mean(predictions, axis=0)
        return np.argmax(final_predictions, axis=1)

    def predict_proba(self, X):
        predictions = []
        for level in range(self.n_levels):
            X_level = self._create_abstraction_features(X, level)
            X_scaled = self.scalers[level].transform(X_level)
            
            level_predictions = []
            for model in self.models[level]:
                level_predictions.append(model.predict_proba(X_scaled))
            
            level_predictions = np.mean(level_predictions, axis=0)
            predictions.append(level_predictions)
        
        final_predictions = np.mean(predictions, axis=0)
        return final_predictions