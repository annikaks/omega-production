import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier

class MultiLevelAbstractionTree(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=3, n_levels=3, min_samples_split=2):
        self.max_depth = max_depth
        self.n_levels = n_levels
        self.min_samples_split = min_samples_split
        self.trees = []
        self.abstractions = []

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Create the initial abstraction (use all features)
        initial_abstraction = np.arange(X.shape[1])
        self.abstractions.append(initial_abstraction)
        
        # Fit trees for each level
        for level in range(self.n_levels):
            # Create a decision tree for this level
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split)
            
            # Select features based on current abstraction
            X_abstracted = X[:, self.abstractions[level]]
            
            # Fit the tree
            tree.fit(X_abstracted, y)
            self.trees.append(tree)
            
            # Create next level abstraction if not at the last level
            if level < self.n_levels - 1:
                next_abstraction = self._create_next_abstraction(tree, self.abstractions[level])
                self.abstractions.append(next_abstraction)
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Make predictions using all trees
        predictions = []
        for level, tree in enumerate(self.trees):
            X_abstracted = X[:, self.abstractions[level]]
            pred = tree.predict(X_abstracted)
            predictions.append(pred)

        # Combine predictions (using majority voting)
        final_predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)),
            axis=0,
            arr=np.array(predictions)
        )

        return final_predictions

    def _create_next_abstraction(self, tree, current_abstraction):
        # Select important features based on feature importances
        importances = tree.feature_importances_
        sorted_idx = np.argsort(importances)
        
        # Select top half of the features
        n_features = len(current_abstraction)
        selected_features = sorted_idx[n_features//2:]
        
        # Map back to original feature indices
        next_abstraction = current_abstraction[selected_features]
        
        return next_abstraction