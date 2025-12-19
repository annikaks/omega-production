import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class ResidualWeightedTreeEnsemble(BaseEstimator, ClassifierMixin):
    """
    Residual-based tree weighting ensemble classifier.
    
    Each subsequent tree receives higher weight if it corrects larger residuals
    from previous trees. Trees that reduce more error get proportionally higher
    weights in the final prediction.
    
    Parameters
    ----------
    n_estimators : int, default=10
        The number of trees in the ensemble.
    
    max_depth : int, default=3
        The maximum depth of each decision tree.
    
    min_samples_split : int, default=2
        The minimum number of samples required to split an internal node.
    
    random_state : int, default=None
        Random state for reproducibility.
    
    Attributes
    ----------
    estimators_ : list
        The collection of fitted sub-estimators.
    
    weights_ : ndarray of shape (n_estimators,)
        The weights assigned to each estimator based on residual correction.
    
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    """
    
    def __init__(self, n_estimators=10, max_depth=3, min_samples_split=2, 
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
    
    def fit(self, X_train, y_train):
        """
        Fit the residual-weighted tree ensemble.
        
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
        
        # Initialize storage
        self.estimators_ = []
        self.weights_ = []
        
        # Convert labels to indices for easier computation
        y_indices = np.searchsorted(self.classes_, y_train)
        
        # Initialize predictions (probabilities for each class)
        ensemble_probs = np.zeros((len(X_train), self.n_classes_))
        
        for i in range(self.n_estimators):
            # Create and fit tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state if self.random_state is None 
                           else self.random_state + i
            )
            tree.fit(X_train, y_train)
            
            # Get predictions from current tree
            tree_probs = tree.predict_proba(X_train)
            
            # Calculate residuals (errors) before adding this tree
            if i == 0:
                # For first tree, all samples have maximum residual
                prev_predictions = np.zeros(len(X_train))
                prev_correct = np.zeros(len(X_train), dtype=bool)
            else:
                prev_predictions = np.argmax(ensemble_probs, axis=1)
                prev_correct = (prev_predictions == y_indices)
            
            # Calculate how much this tree corrects previous errors
            current_predictions = np.argmax(tree_probs, axis=1)
            current_correct = (current_predictions == y_indices)
            
            # Calculate residual correction:
            # - Count samples where previous ensemble was wrong but current tree is right
            # - Weight by confidence of current tree's prediction
            newly_corrected = (~prev_correct) & current_correct
            residual_correction = np.sum(newly_corrected)
            
            # Also consider improvement in probability estimates
            correct_class_probs = tree_probs[np.arange(len(y_train)), y_indices]
            avg_confidence = np.mean(correct_class_probs)
            
            # Weight combines residual correction and confidence
            # Add small epsilon to avoid zero weights
            weight = (residual_correction + 1.0) * (avg_confidence + 0.1)
            
            # Store tree and weight
            self.estimators_.append(tree)
            self.weights_.append(weight)
            
            # Update ensemble predictions with weighted contribution
            if i == 0:
                ensemble_probs = tree_probs * weight
            else:
                ensemble_probs += tree_probs * weight
        
        # Normalize weights
        self.weights_ = np.array(self.weights_)
        if np.sum(self.weights_) > 0:
            self.weights_ = self.weights_ / np.sum(self.weights_)
        else:
            self.weights_ = np.ones(self.n_estimators) / self.n_estimators
        
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
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        # Aggregate weighted predictions
        ensemble_probs = np.zeros((len(X_test), self.n_classes_))
        
        for tree, weight in zip(self.estimators_, self.weights_):
            tree_probs = tree.predict_proba(X_test)
            ensemble_probs += tree_probs * weight
        
        # Normalize probabilities
        row_sums = ensemble_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        ensemble_probs = ensemble_probs / row_sums
        
        return ensemble_probs
    
    def predict(self, X_test):
        """
        Predict class labels for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]