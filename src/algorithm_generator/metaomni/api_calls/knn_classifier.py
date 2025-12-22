import numpy as np
from sklearn.base import BaseEstimator
from collections import Counter


class KNNClassifier(BaseEstimator):
    def __init__(self, n_neighbors=5, metric='euclidean'):
        """
        K-Nearest Neighbors Classifier
        
        Parameters
        ----------
        n_neighbors : int, default=5
            Number of neighbors to use for classification
        metric : str, default='euclidean'
            Distance metric to use ('euclidean' or 'manhattan')
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
    
    def fit(self, X_train, y_train):
        """
        Fit the KNN classifier by storing the training data
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target values
        
        Returns
        -------
        self : object
            Returns self
        """
        self.X_train_ = np.array(X_train)
        self.y_train_ = np.array(y_train)
        self.classes_ = np.unique(y_train)
        return self
    
    def _compute_distance(self, x1, x2):
        """
        Compute distance between two points
        
        Parameters
        ----------
        x1 : array-like
            First point
        x2 : array-like
            Second point
        
        Returns
        -------
        distance : float
            Distance between x1 and x2
        """
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _predict_single(self, x):
        """
        Predict class for a single sample
        
        Parameters
        ----------
        x : array-like of shape (n_features,)
            Single sample
        
        Returns
        -------
        prediction : int or str
            Predicted class label
        """
        # Compute distances to all training samples
        distances = [self._compute_distance(x, x_train) for x_train in self.X_train_]
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.n_neighbors]
        
        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train_[k_indices]
        
        # Return most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        X_test = np.array(X_test)
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)