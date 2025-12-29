import numpy as np
from sklearn.base import BaseEstimator
from collections import Counter


class ManhattanWeightedKNNClassifier(BaseEstimator):
    def __init__(self, n_neighbors=5, epsilon=1e-10):
        """
        K-Nearest Neighbors classifier using Manhattan distance with inverse-distance squared weighting.
        
        Parameters
        ----------
        n_neighbors : int, default=5
            Number of neighbors to use for classification.
        epsilon : float, default=1e-10
            Small constant to avoid division by zero when a test point coincides with a training point.
        """
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier by storing the training data.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples,)
            Target labels.
        
        Returns
        -------
        self : object
            Returns self.
        """
        self.X_train_ = np.array(X_train)
        self.y_train_ = np.array(y_train)
        self.classes_ = np.unique(y_train)
        return self
    
    def _manhattan_distance(self, x1, x2):
        """
        Compute Manhattan distance between two points.
        
        Parameters
        ----------
        x1 : array-like of shape (n_features,)
            First point.
        x2 : array-like of shape (n_features,)
            Second point.
        
        Returns
        -------
        distance : float
            Manhattan distance between x1 and x2.
        """
        return np.sum(np.abs(x1 - x2))
    
    def _predict_single(self, x):
        """
        Predict the class label for a single test point.
        
        Parameters
        ----------
        x : array-like of shape (n_features,)
            Test point.
        
        Returns
        -------
        prediction : scalar
            Predicted class label.
        """
        # Compute Manhattan distances to all training points
        distances = np.array([self._manhattan_distance(x, x_train) for x_train in self.X_train_])
        
        # Get indices of k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:self.n_neighbors]
        
        # Get distances and labels of k nearest neighbors
        k_nearest_distances = distances[k_nearest_indices]
        k_nearest_labels = self.y_train_[k_nearest_indices]
        
        # Compute weights using inverse-distance squared (power of 2)
        # Add epsilon to avoid division by zero
        weights = 1.0 / (k_nearest_distances ** 2 + self.epsilon)
        
        # Compute weighted votes for each class
        class_votes = {}
        for label, weight in zip(k_nearest_labels, weights):
            if label in class_votes:
                class_votes[label] += weight
            else:
                class_votes[label] = weight
        
        # Return class with highest weighted vote
        return max(class_votes, key=class_votes.get)
    
    def predict(self, X_test):
        """
        Predict class labels for test data.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        X_test = np.array(X_test)
        
        # Handle single sample case
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        # Predict for each test point
        predictions = np.array([self._predict_single(x) for x in X_test])
        
        return predictions