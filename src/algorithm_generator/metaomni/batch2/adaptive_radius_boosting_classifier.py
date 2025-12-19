import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import cdist


class AdaptiveRadiusBoostingClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Radius Boosting Classifier.
    
    A boosting classifier that adaptively adjusts the radius of influence for each
    training sample. Misclassified regions get increased radius while correctly
    classified dense areas get decreased radius.
    
    Parameters
    ----------
    n_estimators : int, default=10
        The number of boosting iterations.
    
    initial_radius : float, default=1.0
        The initial radius for all samples.
    
    radius_increase_factor : float, default=1.5
        Factor by which to increase radius for misclassified samples.
    
    radius_decrease_factor : float, default=0.8
        Factor by which to decrease radius for correctly classified samples in dense areas.
    
    density_threshold : float, default=0.5
        Threshold for determining dense areas (fraction of neighbors with same class).
    
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_estimators=10, initial_radius=1.0, 
                 radius_increase_factor=1.5, radius_decrease_factor=0.8,
                 density_threshold=0.5, random_state=None):
        self.n_estimators = n_estimators
        self.initial_radius = initial_radius
        self.radius_increase_factor = radius_increase_factor
        self.radius_decrease_factor = radius_decrease_factor
        self.density_threshold = density_threshold
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Fit the Adaptive Radius Boosting classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        # Store training data
        self.X_ = X.copy()
        self.y_ = y.copy()
        
        # Initialize radii for each sample
        self.radii_ = np.full(len(X), self.initial_radius, dtype=float)
        
        # Store ensemble of radius configurations
        self.ensemble_radii_ = []
        self.ensemble_weights_ = []
        
        # Compute pairwise distances once
        distances = cdist(X, X, metric='euclidean')
        
        # Boosting iterations
        for iteration in range(self.n_estimators):
            # Make predictions with current radii
            predictions = self._predict_with_radii(X, self.radii_)
            
            # Calculate accuracy for weighting
            accuracy = np.mean(predictions == y)
            weight = np.log((accuracy + 1e-10) / (1 - accuracy + 1e-10))
            weight = max(0.0, weight)  # Ensure non-negative
            
            # Store current configuration
            self.ensemble_radii_.append(self.radii_.copy())
            self.ensemble_weights_.append(weight)
            
            # Update radii based on classification results
            self._update_radii(distances, predictions, y)
        
        # Normalize ensemble weights
        total_weight = sum(self.ensemble_weights_)
        if total_weight > 0:
            self.ensemble_weights_ = [w / total_weight for w in self.ensemble_weights_]
        else:
            self.ensemble_weights_ = [1.0 / len(self.ensemble_weights_)] * len(self.ensemble_weights_)
        
        return self
    
    def _predict_with_radii(self, X_test, radii):
        """
        Make predictions using given radii configuration.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        radii : array-like of shape (n_train_samples,)
            Radius for each training sample.
        
        Returns
        -------
        predictions : array of shape (n_samples,)
            Predicted class labels.
        """
        predictions = np.zeros(len(X_test), dtype=self.y_.dtype)
        
        # Compute distances from test samples to training samples
        distances = cdist(X_test, self.X_, metric='euclidean')
        
        for i in range(len(X_test)):
            # Find training samples within their respective radii
            within_radius = distances[i] <= radii
            
            if np.any(within_radius):
                # Get labels of samples within radius
                nearby_labels = self.y_[within_radius]
                nearby_distances = distances[i][within_radius]
                nearby_radii = radii[within_radius]
                
                # Weight by inverse distance and radius
                weights = 1.0 / (nearby_distances + 1e-10) * nearby_radii
                
                # Weighted voting
                class_votes = np.zeros(self.n_classes_)
                for label, weight in zip(nearby_labels, weights):
                    class_idx = np.where(self.classes_ == label)[0][0]
                    class_votes[class_idx] += weight
                
                predictions[i] = self.classes_[np.argmax(class_votes)]
            else:
                # If no samples within radius, use nearest neighbor
                nearest_idx = np.argmin(distances[i])
                predictions[i] = self.y_[nearest_idx]
        
        return predictions
    
    def _update_radii(self, distances, predictions, y_true):
        """
        Update radii based on classification performance and local density.
        
        Parameters
        ----------
        distances : array-like of shape (n_samples, n_samples)
            Pairwise distances between training samples.
        predictions : array-like of shape (n_samples,)
            Current predictions.
        y_true : array-like of shape (n_samples,)
            True labels.
        """
        for i in range(len(self.X_)):
            # Check if sample is misclassified
            if predictions[i] != y_true[i]:
                # Increase radius for misclassified samples
                self.radii_[i] *= self.radius_increase_factor
            else:
                # Check local density for correctly classified samples
                within_radius = distances[i] <= self.radii_[i]
                nearby_labels = y_true[within_radius]
                
                if len(nearby_labels) > 1:
                    # Calculate density (fraction of same-class neighbors)
                    same_class_fraction = np.mean(nearby_labels == y_true[i])
                    
                    # Decrease radius in dense, correctly classified areas
                    if same_class_fraction >= self.density_threshold:
                        self.radii_[i] *= self.radius_decrease_factor
        
        # Ensure radii don't become too small or too large
        self.radii_ = np.clip(self.radii_, self.initial_radius * 0.1, 
                              self.initial_radius * 10.0)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fit has been called
        check_is_fitted(self, ['X_', 'y_', 'ensemble_radii_', 'ensemble_weights_'])
        
        # Validate input
        X = check_array(X)
        
        # Ensemble prediction: weighted voting across all radius configurations
        ensemble_predictions = np.zeros((len(X), self.n_classes_))
        
        for radii, weight in zip(self.ensemble_radii_, self.ensemble_weights_):
            predictions = self._predict_with_radii(X, radii)
            
            for i, pred in enumerate(predictions):
                class_idx = np.where(self.classes_ == pred)[0][0]
                ensemble_predictions[i, class_idx] += weight
        
        # Return class with highest weighted vote
        final_predictions = self.classes_[np.argmax(ensemble_predictions, axis=1)]
        
        return final_predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ['X_', 'y_', 'ensemble_radii_', 'ensemble_weights_'])
        X = check_array(X)
        
        ensemble_predictions = np.zeros((len(X), self.n_classes_))
        
        for radii, weight in zip(self.ensemble_radii_, self.ensemble_weights_):
            predictions = self._predict_with_radii(X, radii)
            
            for i, pred in enumerate(predictions):
                class_idx = np.where(self.classes_ == pred)[0][0]
                ensemble_predictions[i, class_idx] += weight
        
        # Normalize to get probabilities
        row_sums = ensemble_predictions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        proba = ensemble_predictions / row_sums
        
        return proba