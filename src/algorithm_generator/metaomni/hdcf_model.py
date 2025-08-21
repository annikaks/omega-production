import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class HybridDiscreteContForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='auto', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_ = X.shape[1]
        
        # Create a forest of trees
        self.forest_ = []
        
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=self.random_state
            )
            
            # Bootstrap sampling
            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]
            
            # Fit the tree on the bootstrap sample
            tree.fit(X_bootstrap, y_bootstrap)
            
            self.forest_.append(tree)
        
        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        if X.shape[1] != self.n_features_:
            raise ValueError("Number of features in predict does not match number of features in fit")
        
        # Collect predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.forest_])
        
        # Take the majority vote
        final_predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), 
            axis=0, 
            arr=predictions
        )
        
        return self.classes_[final_predictions]

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        if X.shape[1] != self.n_features_:
            raise ValueError("Number of features in predict does not match number of features in fit")
        
        # Collect probabilities from all trees
        probas = np.array([tree.predict_proba(X) for tree in self.forest_])
        
        # Average the probabilities
        final_probas = np.mean(probas, axis=0)
        
        return final_probas