import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class HierarchicalDepthBoostingClassifier(BaseEstimator, ClassifierMixin):
    """
    Hierarchical Depth Boosting Classifier that uses coarse-grained splits in early trees
    and progressively refines with fine-grained splits on residuals.
    
    Parameters
    ----------
    n_estimators : int, default=10
        Number of boosting stages to perform.
    
    min_depth : int, default=1
        Minimum depth for early trees (coarse-grained).
    
    max_depth : int, default=6
        Maximum depth for later trees (fine-grained).
    
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree.
    
    depth_schedule : str, default='linear'
        How to schedule depth progression: 'linear', 'exponential', or 'step'.
    
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_estimators=10, min_depth=1, max_depth=6, 
                 learning_rate=0.1, depth_schedule='linear', random_state=None):
        self.n_estimators = n_estimators
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.depth_schedule = depth_schedule
        self.random_state = random_state
    
    def _get_tree_depth(self, tree_idx):
        """Calculate the depth for a given tree based on the schedule."""
        progress = tree_idx / max(1, self.n_estimators - 1)
        
        if self.depth_schedule == 'linear':
            depth = self.min_depth + progress * (self.max_depth - self.min_depth)
        elif self.depth_schedule == 'exponential':
            depth = self.min_depth * (self.max_depth / self.min_depth) ** progress
        elif self.depth_schedule == 'step':
            # Step function: first half uses min_depth, second half uses max_depth
            depth = self.min_depth if progress < 0.5 else self.max_depth
        else:
            depth = self.min_depth + progress * (self.max_depth - self.min_depth)
        
        return max(1, int(np.round(depth)))
    
    def _convert_to_binary(self, y):
        """Convert multiclass labels to binary format for gradient boosting."""
        n_samples = len(y)
        n_classes = len(self.classes_)
        
        if n_classes == 2:
            # Binary classification: convert to {-1, 1}
            return 2 * (y == self.classes_[1]).astype(float) - 1
        else:
            # Multiclass: one-vs-rest encoding
            y_binary = np.zeros((n_samples, n_classes))
            for i, cls in enumerate(self.classes_):
                y_binary[:, i] = (y == cls).astype(float)
            return y_binary
    
    def _sigmoid(self, x):
        """Numerically stable sigmoid function."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
    
    def fit(self, X_train, y_train):
        """
        Fit the hierarchical depth boosting classifier.
        
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
        self.n_features_in_ = X_train.shape[1]
        
        # Initialize storage for trees
        self.estimators_ = []
        self.tree_depths_ = []
        
        # Convert labels to binary format
        y_binary = self._convert_to_binary(y_train)
        
        if self.n_classes_ == 2:
            # Binary classification
            self._fit_binary(X_train, y_binary)
        else:
            # Multiclass classification (one-vs-rest)
            self._fit_multiclass(X_train, y_binary)
        
        return self
    
    def _fit_binary(self, X, y_binary):
        """Fit binary classification model."""
        n_samples = X.shape[0]
        
        # Initialize predictions with zeros (log-odds)
        F = np.zeros(n_samples)
        
        for i in range(self.n_estimators):
            # Calculate residuals (negative gradient for log loss)
            p = self._sigmoid(F)
            residuals = y_binary - p
            
            # Get depth for this tree
            depth = self._get_tree_depth(i)
            self.tree_depths_.append(depth)
            
            # Fit tree to residuals
            tree = DecisionTreeClassifier(
                max_depth=depth,
                random_state=self.random_state if self.random_state is None 
                           else self.random_state + i
            )
            
            # Convert residuals to binary classes for tree fitting
            # Use sign of residuals as pseudo-labels
            pseudo_labels = (residuals > 0).astype(int)
            tree.fit(X, pseudo_labels)
            
            # Get predictions and convert to residual predictions
            tree_pred = tree.predict_proba(X)[:, 1] * 2 - 1
            
            # Update predictions with learning rate
            F += self.learning_rate * tree_pred
            
            self.estimators_.append(tree)
    
    def _fit_multiclass(self, X, y_binary):
        """Fit multiclass classification model (one-vs-rest)."""
        n_samples = X.shape[0]
        
        # Initialize predictions for each class
        F = np.zeros((n_samples, self.n_classes_))
        
        # Store trees for each class
        self.estimators_ = [[] for _ in range(self.n_classes_)]
        
        for i in range(self.n_estimators):
            depth = self._get_tree_depth(i)
            self.tree_depths_.append(depth)
            
            # Fit one tree per class
            for k in range(self.n_classes_):
                # Calculate residuals for this class
                p = self._sigmoid(F[:, k])
                residuals = y_binary[:, k] - p
                
                # Fit tree to residuals
                tree = DecisionTreeClassifier(
                    max_depth=depth,
                    random_state=self.random_state if self.random_state is None 
                               else self.random_state + i * self.n_classes_ + k
                )
                
                pseudo_labels = (residuals > 0).astype(int)
                tree.fit(X, pseudo_labels)
                
                # Update predictions
                tree_pred = tree.predict_proba(X)[:, 1] * 2 - 1
                F[:, k] += self.learning_rate * tree_pred
                
                self.estimators_[k].append(tree)
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        
        if self.n_classes_ == 2:
            # Binary classification
            F = np.zeros(n_samples)
            
            for tree in self.estimators_:
                tree_pred = tree.predict_proba(X_test)[:, 1] * 2 - 1
                F += self.learning_rate * tree_pred
            
            proba_pos = self._sigmoid(F)
            return np.column_stack([1 - proba_pos, proba_pos])
        else:
            # Multiclass classification
            F = np.zeros((n_samples, self.n_classes_))
            
            for k in range(self.n_classes_):
                for tree in self.estimators_[k]:
                    tree_pred = tree.predict_proba(X_test)[:, 1] * 2 - 1
                    F[:, k] += self.learning_rate * tree_pred
            
            # Apply softmax
            exp_F = np.exp(F - np.max(F, axis=1, keepdims=True))
            return exp_F / np.sum(exp_F, axis=1, keepdims=True)
    
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
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def get_tree_depths(self):
        """Return the depths used for each tree."""
        check_is_fitted(self)
        return self.tree_depths_