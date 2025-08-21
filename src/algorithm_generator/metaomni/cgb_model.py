import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import log_loss
from scipy.stats import entropy

class CompressionGuidedBagger(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=10, max_samples=1.0, max_features=1.0, 
                 base_estimator=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.base_estimator = base_estimator
        self.random_state = random_state
        
    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = np.unique(y)
        
        if self.base_estimator is None:
            self.base_estimator = DecisionTreeClassifier()
        
        # Initialize the bagging classifier
        self.bagger_ = BaggingClassifier(
            base_estimator=self.base_estimator,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            max_features=self.max_features,
            random_state=self.random_state
        )
        
        # Fit the bagging classifier
        self.bagger_.fit(X, y)
        
        # Calculate compression-guided weights
        self.weights_ = self._calculate_weights(X, y)
        
        return self
    
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Get predictions from all estimators
        predictions = np.array([estimator.predict(X) for estimator in self.bagger_.estimators_])
        
        # Apply compression-guided weights
        weighted_predictions = predictions * self.weights_[:, np.newaxis]
        
        # Aggregate predictions
        final_predictions = np.sum(weighted_predictions, axis=0)
        
        return np.argmax(final_predictions, axis=1)
    
    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Get probability predictions from all estimators
        probas = np.array([estimator.predict_proba(X) for estimator in self.bagger_.estimators_])
        
        # Apply compression-guided weights
        weighted_probas = probas * self.weights_[:, np.newaxis, np.newaxis]
        
        # Aggregate probabilities
        final_probas = np.sum(weighted_probas, axis=0)
        
        # Normalize probabilities
        final_probas /= np.sum(final_probas, axis=1)[:, np.newaxis]
        
        return final_probas
    
    def _calculate_weights(self, X, y):
        probas = np.array([estimator.predict_proba(X) for estimator in self.bagger_.estimators_])
        
        # Calculate log-loss for each estimator
        losses = np.array([log_loss(y, proba) for proba in probas])
        
        # Calculate entropy for each estimator's predictions
        entropies = np.array([entropy(proba.T) for proba in probas])
        
        # Combine loss and entropy (you can adjust the balance between these two factors)
        combined_score = losses + 0.1 * entropies
        
        # Convert scores to weights (lower score = higher weight)
        weights = 1 / (combined_score + 1e-10)  # Add small constant to avoid division by zero
        
        # Normalize weights
        weights /= np.sum(weights)
        
        return weights