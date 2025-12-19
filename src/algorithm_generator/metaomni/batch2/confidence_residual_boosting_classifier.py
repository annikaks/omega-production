import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.special import softmax


class ConfidenceResidualBoostingClassifier(BaseEstimator, ClassifierMixin):
    """
    A boosting classifier that builds sequential tree layers where each new tree
    predicts the confidence residuals of previous trees rather than class prediction residuals.
    
    Parameters
    ----------
    n_estimators : int, default=50
        The number of boosting stages to perform.
    
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree.
    
    max_depth : int, default=3
        Maximum depth of the individual trees.
    
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, 
                 min_samples_split=2, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
    
    def _compute_confidence(self, probas):
        """
        Compute confidence scores from probability predictions.
        Uses the maximum probability as confidence.
        """
        return np.max(probas, axis=1)
    
    def _compute_confidence_residuals(self, y_true, current_probas):
        """
        Compute confidence residuals: difference between ideal confidence (1.0 for correct,
        0.0 for incorrect) and current confidence.
        """
        n_samples = len(y_true)
        current_confidence = self._compute_confidence(current_probas)
        
        # Ideal confidence: 1.0 if prediction is correct, 0.0 if incorrect
        current_predictions = np.argmax(current_probas, axis=1)
        ideal_confidence = (current_predictions == y_true).astype(float)
        
        # Residual is the difference between ideal and current confidence
        residuals = ideal_confidence - current_confidence
        
        return residuals
    
    def fit(self, X_train, y_train):
        """
        Fit the confidence residual boosting classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        
        y_train : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_train.shape[1]
        
        # Initialize storage for trees and their weights
        self.trees_ = []
        self.tree_weights_ = []
        
        # Initialize cumulative log-odds for each class
        n_samples = X_train.shape[0]
        self.init_log_odds_ = np.zeros((n_samples, self.n_classes_))
        
        # Set initial uniform probabilities
        for class_idx in range(self.n_classes_):
            class_count = np.sum(y_train == self.classes_[class_idx])
            self.init_log_odds_[:, class_idx] = np.log((class_count + 1) / (n_samples + self.n_classes_))
        
        # Current cumulative predictions (log-odds)
        cumulative_log_odds = self.init_log_odds_.copy()
        
        # Build sequential trees
        for iteration in range(self.n_estimators):
            # Convert log-odds to probabilities
            current_probas = softmax(cumulative_log_odds, axis=1)
            
            # Compute confidence residuals
            confidence_residuals = self._compute_confidence_residuals(y_train, current_probas)
            
            # Train a tree to predict confidence residuals
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state if self.random_state is None else self.random_state + iteration
            )
            
            # Use residuals as weights and fit tree on original targets
            # We fit the tree to predict the class, but weighted by confidence residuals
            sample_weight = np.abs(confidence_residuals) + 1e-10
            tree.fit(X_train, y_train, sample_weight=sample_weight)
            
            # Get tree predictions (probabilities)
            tree_probas = tree.predict_proba(X_train)
            
            # Compute tree's contribution based on confidence residuals
            # Scale the tree's log-odds by the residuals
            tree_log_odds = np.log(tree_probas + 1e-10)
            
            # Weight the contribution by confidence residuals
            residual_weights = confidence_residuals.reshape(-1, 1)
            weighted_tree_contribution = tree_log_odds * np.abs(residual_weights)
            
            # Update cumulative predictions
            cumulative_log_odds += self.learning_rate * weighted_tree_contribution
            
            # Store tree
            self.trees_.append(tree)
            self.tree_weights_.append(self.learning_rate)
        
        return self
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        probas : array of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        
        # Initialize with mean log-odds from training
        cumulative_log_odds = np.tile(np.mean(self.init_log_odds_, axis=0), (n_samples, 1))
        
        # Accumulate predictions from all trees
        for tree_idx, tree in enumerate(self.trees_):
            tree_probas = tree.predict_proba(X_test)
            tree_log_odds = np.log(tree_probas + 1e-10)
            
            # Add weighted contribution
            cumulative_log_odds += self.tree_weights_[tree_idx] * tree_log_odds
        
        # Convert to probabilities
        probas = softmax(cumulative_log_odds, axis=1)
        
        return probas
    
    def predict(self, X_test):
        """
        Predict class labels for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        probas = self.predict_proba(X_test)
        return self.classes_[np.argmax(probas, axis=1)]