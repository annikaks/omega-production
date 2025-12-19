import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.linear_model import Perceptron


class WeightedPerceptronEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble of perceptrons with different random initializations combined via
    bias-variance optimal weighted voting.
    
    Parameters
    ----------
    n_estimators : int, default=10
        Number of perceptrons in the ensemble.
    max_iter : int, default=1000
        Maximum number of passes over the training data.
    tol : float, default=1e-3
        Tolerance for stopping criterion.
    eta0 : float, default=1.0
        Learning rate for perceptrons.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_estimators=10, max_iter=1000, tol=1e-3, eta0=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.tol = tol
        self.eta0 = eta0
        self.random_state = random_state
    
    def fit(self, X_train, y_train):
        """
        Fit the ensemble of perceptrons and compute optimal weights.
        
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
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Train ensemble of perceptrons with different random seeds
        self.estimators_ = []
        for i in range(self.n_estimators):
            seed = rng.randint(0, 2**31 - 1)
            perceptron = Perceptron(
                max_iter=self.max_iter,
                tol=self.tol,
                eta0=self.eta0,
                random_state=seed,
                shuffle=True
            )
            perceptron.fit(X_train, y_train)
            self.estimators_.append(perceptron)
        
        # Compute optimal weights based on training performance
        self.weights_ = self._compute_optimal_weights(X_train, y_train)
        
        return self
    
    def _compute_optimal_weights(self, X, y):
        """
        Compute bias-variance optimal weights for ensemble members.
        
        Uses inverse error rate as a proxy for optimal weighting.
        Better performing models get higher weights.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Validation data.
        y : array-like of shape (n_samples,)
            True labels.
        
        Returns
        -------
        weights : ndarray of shape (n_estimators,)
            Normalized weights for each estimator.
        """
        weights = np.zeros(self.n_estimators)
        
        for i, estimator in enumerate(self.estimators_):
            # Predict on training data
            y_pred = estimator.predict(X)
            
            # Compute accuracy
            accuracy = np.mean(y_pred == y)
            
            # Use accuracy as weight (with small epsilon to avoid division by zero)
            # Higher accuracy -> higher weight
            weights[i] = max(accuracy, 1e-10)
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        return weights
    
    def predict(self, X_test):
        """
        Predict class labels using weighted voting.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fitted
        check_is_fitted(self, ['estimators_', 'weights_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Get predictions from all estimators
        predictions = np.array([estimator.predict(X_test) for estimator in self.estimators_])
        
        # Perform weighted voting
        n_samples = X_test.shape[0]
        y_pred = np.zeros(n_samples, dtype=self.classes_.dtype)
        
        for i in range(n_samples):
            # Get predictions for this sample from all estimators
            sample_predictions = predictions[:, i]
            
            # Compute weighted votes for each class
            class_votes = {}
            for class_label in self.classes_:
                # Sum weights of estimators that predicted this class
                mask = sample_predictions == class_label
                class_votes[class_label] = np.sum(self.weights_[mask])
            
            # Select class with highest weighted vote
            y_pred[i] = max(class_votes, key=class_votes.get)
        
        return y_pred
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities using weighted voting.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        # Check if fitted
        check_is_fitted(self, ['estimators_', 'weights_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Get predictions from all estimators
        predictions = np.array([estimator.predict(X_test) for estimator in self.estimators_])
        
        # Compute weighted probabilities
        n_samples = X_test.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))
        
        for i in range(n_samples):
            # Get predictions for this sample from all estimators
            sample_predictions = predictions[:, i]
            
            # Compute weighted votes for each class
            for j, class_label in enumerate(self.classes_):
                mask = sample_predictions == class_label
                proba[i, j] = np.sum(self.weights_[mask])
        
        return proba