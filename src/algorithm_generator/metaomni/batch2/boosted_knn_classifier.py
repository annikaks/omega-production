import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from collections import Counter


class BoostedKNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Boosted KNN Classifier that iteratively trains on residual errors.
    
    Parameters
    ----------
    n_estimators : int, default=10
        Number of boosting iterations
    k_neighbors : int, default=5
        Number of neighbors to use for KNN
    learning_rate : float, default=1.0
        Shrinkage parameter for each estimator's contribution
    """
    
    def __init__(self, n_estimators=10, k_neighbors=5, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.k_neighbors = k_neighbors
        self.learning_rate = learning_rate
    
    def fit(self, X_train, y_train):
        """
        Fit the boosted KNN classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
        """
        X_train, y_train = check_X_y(X_train, y_train)
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        
        # Store training data and initialize weights
        self.X_train_ = X_train.copy()
        self.y_train_ = y_train.copy()
        self.n_samples_ = X_train.shape[0]
        
        # Initialize sample weights uniformly
        self.sample_weights_ = np.ones(self.n_samples_) / self.n_samples_
        
        # Store estimators and their weights
        self.estimators_ = []
        self.estimator_weights_ = []
        
        # For multi-class, use one-vs-rest approach
        if self.n_classes_ == 2:
            self._fit_binary(X_train, y_train)
        else:
            self._fit_multiclass(X_train, y_train)
        
        return self
    
    def _fit_binary(self, X_train, y_train):
        """Fit binary classification."""
        # Convert labels to -1, 1
        y_binary = np.where(y_train == self.classes_[0], -1, 1)
        
        # Initialize weights
        weights = np.ones(self.n_samples_) / self.n_samples_
        
        for m in range(self.n_estimators):
            # Store current training data and weights
            estimator = {
                'X': X_train.copy(),
                'y': y_binary.copy(),
                'weights': weights.copy()
            }
            
            # Make predictions using weighted KNN
            predictions = self._weighted_knn_predict(
                X_train, X_train, y_binary, weights
            )
            
            # Calculate weighted error
            incorrect = predictions != y_binary
            error = np.sum(weights[incorrect]) / np.sum(weights)
            
            # Avoid division by zero and perfect classification
            error = np.clip(error, 1e-10, 1 - 1e-10)
            
            # Calculate estimator weight (AdaBoost formula)
            alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
            
            # Update sample weights
            weights *= np.exp(-alpha * y_binary * predictions)
            weights /= np.sum(weights)  # Normalize
            
            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)
    
    def _fit_multiclass(self, X_train, y_train):
        """Fit multi-class classification using SAMME algorithm."""
        weights = np.ones(self.n_samples_) / self.n_samples_
        
        for m in range(self.n_estimators):
            # Store current training data and weights
            estimator = {
                'X': X_train.copy(),
                'y': y_train.copy(),
                'weights': weights.copy()
            }
            
            # Make predictions using weighted KNN
            predictions = self._weighted_knn_predict_multiclass(
                X_train, X_train, y_train, weights
            )
            
            # Calculate weighted error
            incorrect = predictions != y_train
            error = np.sum(weights[incorrect]) / np.sum(weights)
            
            # Avoid division by zero and perfect classification
            error = np.clip(error, 1e-10, 1 - 1e-10)
            
            # Calculate estimator weight (SAMME formula)
            alpha = self.learning_rate * (
                np.log((1 - error) / error) + np.log(self.n_classes_ - 1)
            )
            
            # Update sample weights
            weights[incorrect] *= np.exp(alpha)
            weights /= np.sum(weights)  # Normalize
            
            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)
    
    def _weighted_knn_predict(self, X_test, X_train, y_train, weights):
        """Predict using weighted KNN for binary classification."""
        predictions = np.zeros(len(X_test))
        
        for i, x in enumerate(X_test):
            # Calculate weighted distances
            distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
            
            # Get k nearest neighbors
            k = min(self.k_neighbors, len(X_train))
            nearest_indices = np.argpartition(distances, k-1)[:k]
            
            # Weight votes by both distance and sample weight
            neighbor_weights = weights[nearest_indices]
            neighbor_labels = y_train[nearest_indices]
            neighbor_distances = distances[nearest_indices]
            
            # Avoid division by zero in distance weighting
            distance_weights = 1.0 / (neighbor_distances + 1e-10)
            combined_weights = neighbor_weights * distance_weights
            
            # Weighted vote
            vote = np.sum(combined_weights * neighbor_labels)
            predictions[i] = 1 if vote > 0 else -1
        
        return predictions
    
    def _weighted_knn_predict_multiclass(self, X_test, X_train, y_train, weights):
        """Predict using weighted KNN for multi-class classification."""
        predictions = np.zeros(len(X_test), dtype=y_train.dtype)
        
        for i, x in enumerate(X_test):
            # Calculate weighted distances
            distances = np.sqrt(np.sum((X_train - x) ** 2, axis=1))
            
            # Get k nearest neighbors
            k = min(self.k_neighbors, len(X_train))
            nearest_indices = np.argpartition(distances, k-1)[:k]
            
            # Weight votes by both distance and sample weight
            neighbor_weights = weights[nearest_indices]
            neighbor_labels = y_train[nearest_indices]
            neighbor_distances = distances[nearest_indices]
            
            # Avoid division by zero in distance weighting
            distance_weights = 1.0 / (neighbor_distances + 1e-10)
            combined_weights = neighbor_weights * distance_weights
            
            # Weighted vote for each class
            class_votes = {}
            for label, weight in zip(neighbor_labels, combined_weights):
                class_votes[label] = class_votes.get(label, 0) + weight
            
            predictions[i] = max(class_votes, key=class_votes.get)
        
        return predictions
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        if self.n_classes_ == 2:
            return self._predict_binary(X_test)
        else:
            return self._predict_multiclass(X_test)
    
    def _predict_binary(self, X_test):
        """Predict for binary classification."""
        # Aggregate predictions from all estimators
        final_predictions = np.zeros(len(X_test))
        
        for estimator, alpha in zip(self.estimators_, self.estimator_weights_):
            predictions = self._weighted_knn_predict(
                X_test,
                estimator['X'],
                estimator['y'],
                estimator['weights']
            )
            final_predictions += alpha * predictions
        
        # Convert back to original labels
        y_pred = np.where(final_predictions > 0, self.classes_[1], self.classes_[0])
        return y_pred
    
    def _predict_multiclass(self, X_test):
        """Predict for multi-class classification."""
        # Aggregate predictions from all estimators
        class_scores = {c: np.zeros(len(X_test)) for c in self.classes_}
        
        for estimator, alpha in zip(self.estimators_, self.estimator_weights_):
            predictions = self._weighted_knn_predict_multiclass(
                X_test,
                estimator['X'],
                estimator['y'],
                estimator['weights']
            )
            
            # Add weighted votes for each class
            for i, pred in enumerate(predictions):
                class_scores[pred][i] += alpha
        
        # Get class with highest score for each sample
        y_pred = np.array([
            max(self.classes_, key=lambda c: class_scores[c][i])
            for i in range(len(X_test))
        ])
        
        return y_pred