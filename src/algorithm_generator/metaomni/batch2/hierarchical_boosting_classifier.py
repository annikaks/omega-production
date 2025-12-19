import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class HierarchicalBoostingClassifier(BaseEstimator, ClassifierMixin):
    """
    Hierarchical Boosting Classifier that captures coarse patterns in early iterations
    and refines fine-grained decision boundaries in later iterations.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of boosting iterations.
    
    n_levels : int, default=3
        Number of hierarchical levels (abstraction levels).
    
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each classifier.
    
    max_depth_schedule : str or list, default='progressive'
        Strategy for tree depth across iterations:
        - 'progressive': gradually increase depth from shallow to deep
        - list: custom list of max_depths for each level
    
    min_samples_split_schedule : str or list, default='progressive'
        Strategy for min_samples_split across iterations:
        - 'progressive': gradually decrease from high to low
        - list: custom list of min_samples_split for each level
    
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(
        self,
        n_estimators=100,
        n_levels=3,
        learning_rate=0.1,
        max_depth_schedule='progressive',
        min_samples_split_schedule='progressive',
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.n_levels = n_levels
        self.learning_rate = learning_rate
        self.max_depth_schedule = max_depth_schedule
        self.min_samples_split_schedule = min_samples_split_schedule
        self.random_state = random_state
    
    def _get_tree_params(self, iteration):
        """
        Get tree parameters for a given iteration based on hierarchical level.
        Early iterations use shallow trees (coarse patterns),
        later iterations use deeper trees (fine-grained patterns).
        """
        # Determine which level this iteration belongs to
        estimators_per_level = self.n_estimators // self.n_levels
        level = min(iteration // estimators_per_level, self.n_levels - 1)
        
        # Progressive depth schedule
        if isinstance(self.max_depth_schedule, str) and self.max_depth_schedule == 'progressive':
            # Start with depth 1-2 (stumps/shallow), end with depth 6-8 (deeper)
            min_depth = 1
            max_depth_final = 8
            max_depth = min_depth + int((max_depth_final - min_depth) * level / (self.n_levels - 1))
        elif isinstance(self.max_depth_schedule, list):
            max_depth = self.max_depth_schedule[level]
        else:
            max_depth = 3
        
        # Progressive min_samples_split schedule
        if isinstance(self.min_samples_split_schedule, str) and self.min_samples_split_schedule == 'progressive':
            # Start with high min_samples_split (coarse), end with low (fine-grained)
            max_samples = 50
            min_samples = 2
            min_samples_split = max_samples - int((max_samples - min_samples) * level / (self.n_levels - 1))
        elif isinstance(self.min_samples_split_schedule, list):
            min_samples_split = self.min_samples_split_schedule[level]
        else:
            min_samples_split = 20
        
        return {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'level': level
        }
    
    def fit(self, X_train, y_train):
        """
        Fit the hierarchical boosting classifier.
        
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
        
        # Initialize
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_levels_ = []
        
        # Convert labels to {-1, 1} for binary classification
        if self.n_classes_ == 2:
            y_encoded = np.where(y_train == self.classes_[0], -1, 1)
        else:
            # For multiclass, use one-vs-rest approach
            y_encoded = y_train
        
        # Initialize predictions (for binary: margin, for multiclass: probabilities)
        if self.n_classes_ == 2:
            F = np.zeros(len(y_train))
        else:
            F = np.zeros((len(y_train), self.n_classes_))
        
        # Boosting iterations
        for iteration in range(self.n_estimators):
            # Get tree parameters for this iteration (hierarchical level)
            tree_params = self._get_tree_params(iteration)
            
            if self.n_classes_ == 2:
                # Binary classification with AdaBoost-style weighting
                # Compute sample weights based on current predictions
                sample_weight = np.exp(-y_encoded * F)
                sample_weight /= sample_weight.sum()
                
                # Fit weak learner
                estimator = DecisionTreeClassifier(
                    max_depth=tree_params['max_depth'],
                    min_samples_split=tree_params['min_samples_split'],
                    random_state=self.random_state
                )
                estimator.fit(X_train, y_train, sample_weight=sample_weight)
                
                # Predict
                y_pred = estimator.predict(X_train)
                y_pred_encoded = np.where(y_pred == self.classes_[0], -1, 1)
                
                # Compute weighted error
                incorrect = y_pred_encoded != y_encoded
                error = np.sum(sample_weight * incorrect)
                
                # Avoid division by zero
                error = np.clip(error, 1e-10, 1 - 1e-10)
                
                # Compute estimator weight
                alpha = 0.5 * np.log((1 - error) / error)
                
                # Update predictions
                F += self.learning_rate * alpha * y_pred_encoded
                
                # Store estimator
                self.estimators_.append(estimator)
                self.estimator_weights_.append(self.learning_rate * alpha)
                self.estimator_levels_.append(tree_params['level'])
                
            else:
                # Multiclass classification with SAMME.R
                estimator = DecisionTreeClassifier(
                    max_depth=tree_params['max_depth'],
                    min_samples_split=tree_params['min_samples_split'],
                    random_state=self.random_state
                )
                
                # Compute sample weights
                if iteration == 0:
                    sample_weight = np.ones(len(y_train)) / len(y_train)
                else:
                    # Compute probabilities from current F
                    probs = np.exp(F - F.max(axis=1, keepdims=True))
                    probs /= probs.sum(axis=1, keepdims=True)
                    
                    # Weight samples by current error
                    y_one_hot = np.zeros((len(y_train), self.n_classes_))
                    for i, cls in enumerate(self.classes_):
                        y_one_hot[y_train == cls, i] = 1
                    
                    sample_weight = 1 - (probs * y_one_hot).sum(axis=1)
                    sample_weight = np.clip(sample_weight, 1e-10, None)
                    sample_weight /= sample_weight.sum()
                
                estimator.fit(X_train, y_train, sample_weight=sample_weight)
                
                # Get probability predictions
                proba = estimator.predict_proba(X_train)
                proba = np.clip(proba, 1e-10, 1 - 1e-10)
                
                # Update F (log-odds)
                log_proba = np.log(proba)
                F += self.learning_rate * (self.n_classes_ - 1) / self.n_classes_ * log_proba
                
                # Store estimator
                self.estimators_.append(estimator)
                self.estimator_weights_.append(self.learning_rate)
                self.estimator_levels_.append(tree_params['level'])
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        if self.n_classes_ == 2:
            # Binary classification
            F = np.zeros(len(X_test))
            
            for estimator, weight in zip(self.estimators_, self.estimator_weights_):
                y_pred = estimator.predict(X_test)
                y_pred_encoded = np.where(y_pred == self.classes_[0], -1, 1)
                F += weight * y_pred_encoded
            
            return np.where(F <= 0, self.classes_[0], self.classes_[1])
        
        else:
            # Multiclass classification
            F = np.zeros((len(X_test), self.n_classes_))
            
            for estimator, weight in zip(self.estimators_, self.estimator_weights_):
                proba = estimator.predict_proba(X_test)
                proba = np.clip(proba, 1e-10, 1 - 1e-10)
                log_proba = np.log(proba)
                F += weight * (self.n_classes_ - 1) / self.n_classes_ * log_proba
            
            return self.classes_[np.argmax(F, axis=1)]
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        if self.n_classes_ == 2:
            # Binary classification
            F = np.zeros(len(X_test))
            
            for estimator, weight in zip(self.estimators_, self.estimator_weights_):
                y_pred = estimator.predict(X_test)
                y_pred_encoded = np.where(y_pred == self.classes_[0], -1, 1)
                F += weight * y_pred_encoded
            
            # Convert to probabilities
            proba_class1 = 1 / (1 + np.exp(-2 * F))
            return np.vstack([1 - proba_class1, proba_class1]).T
        
        else:
            # Multiclass classification
            F = np.zeros((len(X_test), self.n_classes_))
            
            for estimator, weight in zip(self.estimators_, self.estimator_weights_):
                proba = estimator.predict_proba(X_test)
                proba = np.clip(proba, 1e-10, 1 - 1e-10)
                log_proba = np.log(proba)
                F += weight * (self.n_classes_ - 1) / self.n_classes_ * log_proba
            
            # Convert to probabilities
            proba = np.exp(F - F.max(axis=1, keepdims=True))
            proba /= proba.sum(axis=1, keepdims=True)
            return proba