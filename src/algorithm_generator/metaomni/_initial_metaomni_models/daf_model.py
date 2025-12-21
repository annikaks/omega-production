import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class DimensionalityAwareForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_features='sqrt', max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_ = X.shape[1]
        
        # Initialize the forest
        self.estimators_ = []
        
        # Generate random subspaces
        np.random.seed(self.random_state)
        self.subspaces_ = [np.random.choice(self.n_features_, 
                                            size=int(np.sqrt(self.n_features_)), 
                                            replace=False) 
                           for _ in range(self.n_estimators)]
        
        # Train each tree on its subspace
        for subspace in self.subspaces_:
            tree = DecisionTreeClassifier(
                max_features=self.max_features,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            tree.fit(X[:, subspace], y)
            self.estimators_.append(tree)
        
        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Ensure X has the correct number of features
        if X.shape[1] != self.n_features_:
            raise ValueError("Number of features in X does not match training data.")
        
        # Collect predictions from all trees
        predictions = []
        for tree, subspace in zip(self.estimators_, self.subspaces_):
            predictions.append(tree.predict(X[:, subspace]))
        
        # Majority voting
        predictions = np.array(predictions)
        maj = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, minlength=len(self.classes_))),
            axis=0,
            arr=predictions)
        
        return self.classes_[maj]