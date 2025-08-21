import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class DirectionalTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        self.direction = None

    def fit(self, X, y):
        self.direction = np.random.randn(X.shape[1])
        self.direction /= np.linalg.norm(self.direction)
        X_proj = X.dot(self.direction)
        self.tree.fit(X_proj.reshape(-1, 1), y)

    def predict(self, X):
        X_proj = X.dot(self.direction)
        return self.tree.predict(X_proj.reshape(-1, 1))

class DirectionalEnsembleTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Create and fit the trees
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DirectionalTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X, y)
            self.trees.append(tree)

        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Make predictions with each tree
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Return the majority vote
        return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=predictions)

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Make predictions with each tree
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Calculate class probabilities
        probas = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(self.classes_)) / len(self.trees), 
                                     axis=0, arr=predictions)
        
        return probas.T