import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class MultiGrainBoostingClassifier(BaseEstimator, ClassifierMixin):
    """
    Multi-Grain Boosting Classifier that progressively refines decision boundaries.
    
    Early iterations use shallow trees (coarse grain) to establish broad decision
    boundaries, while later iterations use deeper trees (fine grain) to refine
    predictions in misclassified regions.
    
    Parameters
    ----------
    n_estimators : int, default=50
        The number of boosting iterations.
    learning_rate : float, default=1.0
        Learning rate shrinks the contribution of each classifier.
    min_depth : int, default=1
        Minimum depth for decision trees in early iterations.
    max_depth : int, default=6
        Maximum depth for decision trees in later iterations.
    grain_transition : float, default=0.5
        Fraction of iterations at which to transition from coarse to fine grain.
        Value between 0 and 1.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_estimators=50, learning_rate=1.0, min_depth=1,
                 max_depth=6, grain_transition=0.5, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.grain_transition = grain_transition
        self.random_state = random_state
    
    def _get_tree_depth(self, iteration):
        """
        Calculate tree depth for current iteration using progressive deepening.
        
        Parameters
        ----------
        iteration : int
            Current boosting iteration (0-indexed).
            
        Returns
        -------
        int
            Depth for the current tree.
        """
        transition_point = int(self.n_estimators * self.grain_transition)
        
        if iteration < transition_point:
            # Coarse grain phase: gradually increase from min_depth
            progress = iteration / max(transition_point, 1)
            mid_depth = (self.min_depth + self.max_depth) // 2
            depth = int(self.min_depth + progress * (mid_depth - self.min_depth))
        else:
            # Fine grain phase: gradually increase to max_depth
            progress = (iteration - transition_point) / max(self.n_estimators - transition_point, 1)
            mid_depth = (self.min_depth + self.max_depth) // 2
            depth = int(mid_depth + progress * (self.max_depth - mid_depth))
        
        return max(self.min_depth, min(depth, self.max_depth))
    
    def _compute_sample_weights(self, y_true, y_pred, sample_weight, iteration):
        """
        Compute updated sample weights with focus on misclassified regions.
        
        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.
        sample_weight : array-like
            Current sample weights.
        iteration : int
            Current iteration number.
            
        Returns
        -------
        array-like
            Updated sample weights.
        """
        # Identify misclassified samples
        incorrect = (y_true != y_pred).astype(float)
        
        # Calculate error rate
        error = np.sum(sample_weight * incorrect) / np.sum(sample_weight)
        error = np.clip(error, 1e-10, 1 - 1e-10)
        
        # Calculate alpha (estimator weight)
        alpha = self.learning_rate * 0.5 * np.log((1 - error) / error)
        
        # Update sample weights - increase for misclassified samples
        sample_weight *= np.exp(alpha * incorrect)
        
        # Normalize weights
        sample_weight /= np.sum(sample_weight)
        
        # In fine-grain phase, further boost weights in misclassified regions
        transition_point = int(self.n_estimators * self.grain_transition)
        if iteration >= transition_point:
            # Apply additional focus on persistent errors
            boost_factor = 1.0 + (iteration - transition_point) / (self.n_estimators - transition_point)
            sample_weight[incorrect.astype(bool)] *= boost_factor
            sample_weight /= np.sum(sample_weight)
        
        return sample_weight, alpha
    
    def fit(self, X_train, y_train):
        """
        Fit the Multi-Grain Boosting classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self
            Fitted estimator.
        """
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_train.shape[1]
        
        # Initialize
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_depths_ = []
        
        # Initialize sample weights uniformly
        n_samples = X_train.shape[0]
        sample_weight = np.ones(n_samples) / n_samples
        
        # Boosting iterations
        for iteration in range(self.n_estimators):
            # Determine tree depth for this iteration
            depth = self._get_tree_depth(iteration)
            self.estimator_depths_.append(depth)
            
            # Create and fit weak learner
            estimator = DecisionTreeClassifier(
                max_depth=depth,
                random_state=self.random_state if self.random_state is None 
                           else self.random_state + iteration,
                splitter='best'
            )
            
            estimator.fit(X_train, y_train, sample_weight=sample_weight)
            
            # Make predictions
            y_pred = estimator.predict(X_train)
            
            # Update sample weights and calculate estimator weight
            sample_weight, alpha = self._compute_sample_weights(
                y_train, y_pred, sample_weight, iteration
            )
            
            # Store estimator and its weight
            self.estimators_.append(estimator)
            self.estimator_weights_.append(alpha)
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns
        -------
        array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        # Get weighted predictions from all estimators
        predictions = np.zeros((X_test.shape[0], self.n_classes_))
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            # Get predictions from current estimator
            pred = estimator.predict(X_test)
            
            # Add weighted vote for each class
            for i, class_label in enumerate(self.classes_):
                predictions[:, i] += weight * (pred == class_label)
        
        # Return class with highest weighted vote
        return self.classes_[np.argmax(predictions, axis=1)]
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
            
        Returns
        -------
        array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        # Get weighted predictions from all estimators
        predictions = np.zeros((X_test.shape[0], self.n_classes_))
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            pred = estimator.predict(X_test)
            
            for i, class_label in enumerate(self.classes_):
                predictions[:, i] += weight * (pred == class_label)
        
        # Normalize to get probabilities
        predictions /= np.sum(predictions, axis=1, keepdims=True)
        
        return predictions