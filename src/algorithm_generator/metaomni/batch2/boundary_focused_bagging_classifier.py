import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class BoundaryFocusedBaggingClassifier(BaseEstimator, ClassifierMixin):
    """
    Bagging classifier with dynamic weights that increase sampling probability
    for instances near decision boundaries identified by previous trees.
    
    Parameters
    ----------
    base_estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a DecisionTreeClassifier.
    
    n_estimators : int, default=10
        The number of base estimators in the ensemble.
    
    max_samples : float or int, default=1.0
        The number of samples to draw to train each base estimator.
        If float, then draw max_samples * n_samples samples.
    
    boundary_threshold : float, default=0.3
        Threshold for identifying boundary instances. Instances with prediction
        uncertainty (distance from 0.5 probability) below this are considered
        near the boundary.
    
    weight_amplification : float, default=2.0
        Factor by which to amplify weights for boundary instances.
    
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0,
                 boundary_threshold=0.3, weight_amplification=2.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.boundary_threshold = boundary_threshold
        self.weight_amplification = weight_amplification
        self.random_state = random_state
    
    def _initialize_weights(self, n_samples):
        """Initialize uniform weights for all samples."""
        return np.ones(n_samples) / n_samples
    
    def _update_weights(self, X, y, estimator):
        """
        Update sample weights based on proximity to decision boundary.
        
        Instances with prediction probabilities close to 0.5 (uncertain)
        are considered near the boundary and receive higher weights.
        """
        n_samples = X.shape[0]
        
        # Get prediction probabilities
        if hasattr(estimator, 'predict_proba'):
            proba = estimator.predict_proba(X)
            # For binary classification, use the probability of positive class
            if proba.shape[1] == 2:
                uncertainty = np.abs(proba[:, 1] - 0.5)
            else:
                # For multiclass, use entropy-based uncertainty
                uncertainty = -np.sum(proba * np.log(proba + 1e-10), axis=1)
                uncertainty = 1 - (uncertainty / np.log(proba.shape[1]))
        else:
            # If no predict_proba, use prediction correctness
            predictions = estimator.predict(X)
            uncertainty = (predictions != y).astype(float)
        
        # Identify boundary instances (high uncertainty)
        if hasattr(estimator, 'predict_proba') and proba.shape[1] == 2:
            is_boundary = uncertainty < self.boundary_threshold
        else:
            is_boundary = uncertainty > 0.5
        
        # Update weights
        weights = np.ones(n_samples)
        weights[is_boundary] *= self.weight_amplification
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return weights
    
    def _sample_with_weights(self, X, y, weights, rng):
        """Sample instances according to weights."""
        n_samples = X.shape[0]
        
        if isinstance(self.max_samples, float):
            n_samples_bootstrap = int(self.max_samples * n_samples)
        else:
            n_samples_bootstrap = self.max_samples
        
        # Sample indices according to weights
        indices = rng.choice(n_samples, size=n_samples_bootstrap, 
                            replace=True, p=weights)
        
        return X[indices], y[indices]
    
    def fit(self, X_train, y_train):
        """
        Fit the boundary-focused bagging classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        
        y_train : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        check_classification_targets(y_train)
        
        self.classes_ = np.unique(y_train)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_train.shape[1]
        
        # Initialize base estimator
        if self.base_estimator is None:
            base_est = DecisionTreeClassifier()
        else:
            base_est = self.base_estimator
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Initialize estimators list
        self.estimators_ = []
        
        # Initialize weights uniformly
        weights = self._initialize_weights(X_train.shape[0])
        
        # Train estimators sequentially with dynamic weight updates
        for i in range(self.n_estimators):
            # Clone base estimator
            estimator = clone(base_est)
            
            # Sample training data according to current weights
            X_sample, y_sample = self._sample_with_weights(X_train, y_train, 
                                                           weights, rng)
            
            # Fit estimator
            estimator.fit(X_sample, y_sample)
            
            # Store estimator
            self.estimators_.append(estimator)
            
            # Update weights based on boundary proximity
            # (for next iteration, except the last one)
            if i < self.n_estimators - 1:
                weights = self._update_weights(X_train, y_train, estimator)
        
        return self
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        # Aggregate predictions from all estimators
        all_proba = np.zeros((X_test.shape[0], self.n_classes_))
        
        for estimator in self.estimators_:
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X_test)
            else:
                # For estimators without predict_proba, use hard predictions
                predictions = estimator.predict(X_test)
                proba = np.zeros((X_test.shape[0], self.n_classes_))
                for i, cls in enumerate(self.classes_):
                    proba[:, i] = (predictions == cls).astype(float)
            
            all_proba += proba
        
        # Average probabilities
        all_proba /= len(self.estimators_)
        
        return all_proba
    
    def predict(self, X_test):
        """
        Predict class labels for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]