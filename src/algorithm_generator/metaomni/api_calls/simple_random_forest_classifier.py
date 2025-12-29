from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from collections import Counter


class SimpleRandomForestClassifier(BaseEstimator):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 max_features='sqrt', bootstrap=True, random_state=None):
        """
        Simple Random Forest Classifier implementation.
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of trees in the forest
        max_depth : int, default=None
            Maximum depth of each tree
        min_samples_split : int, default=2
            Minimum samples required to split a node
        max_features : str or int, default='sqrt'
            Number of features to consider for best split ('sqrt', 'log2', or int)
        bootstrap : bool, default=True
            Whether to use bootstrap samples
        random_state : int, default=None
            Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
    def fit(self, X_train, y_train):
        """
        Fit the random forest classifier.
        
        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : object
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.n_samples_, self.n_features_ = X_train.shape
        self.classes_ = np.unique(y_train)
        self.n_classes_ = len(self.classes_)
        
        # Determine max_features
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(self.n_features_))
        elif self.max_features == 'log2':
            max_features = int(np.log2(self.n_features_))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = self.n_features_
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Train individual trees
        self.trees_ = []
        for i in range(self.n_estimators):
            # Create bootstrap sample
            if self.bootstrap:
                indices = rng.choice(self.n_samples_, size=self.n_samples_, replace=True)
                X_sample = X_train[indices]
                y_sample = y_train[indices]
            else:
                X_sample = X_train
                y_sample = y_train
            
            # Create and train decision tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                random_state=rng.randint(0, 10000)
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
        
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        X_test = np.array(X_test)
        
        # Collect predictions from all trees
        predictions = np.array([tree.predict(X_test) for tree in self.trees_])
        
        # Majority voting
        y_pred = np.array([
            Counter(predictions[:, i]).most_common(1)[0][0] 
            for i in range(X_test.shape[0])
        ])
        
        return y_pred
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
        
        Returns:
        --------
        proba : array of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        X_test = np.array(X_test)
        
        # Collect probability predictions from all trees
        all_probas = np.array([tree.predict_proba(X_test) for tree in self.trees_])
        
        # Average probabilities across all trees
        avg_probas = np.mean(all_probas, axis=0)
        
        return avg_probas