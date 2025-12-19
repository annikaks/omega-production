import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class GradientBoostedForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Gradient Boosting on Random Forest Residuals Classifier.
    
    Trains subsequent random forests on prediction errors with exponentially
    decaying learning rates.
    
    Parameters
    ----------
    n_estimators : int, default=10
        Number of boosting stages (random forests) to train.
    
    initial_learning_rate : float, default=1.0
        Initial learning rate for the first forest.
    
    decay_rate : float, default=0.9
        Exponential decay rate for learning rates across stages.
    
    n_trees_per_forest : int, default=100
        Number of trees in each random forest.
    
    max_depth : int, default=None
        Maximum depth of trees in random forests.
    
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(
        self,
        n_estimators=10,
        initial_learning_rate=1.0,
        decay_rate=0.9,
        n_trees_per_forest=100,
        max_depth=None,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        self.n_trees_per_forest = n_trees_per_forest
        self.max_depth = max_depth
        self.random_state = random_state
    
    def _compute_learning_rate(self, stage):
        """Compute exponentially decaying learning rate for a given stage."""
        return self.initial_learning_rate * (self.decay_rate ** stage)
    
    def fit(self, X_train, y_train):
        """
        Fit the gradient boosted forest classifier.
        
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
        
        # Store forests and learning rates
        self.forests_ = []
        self.learning_rates_ = []
        
        # Initialize predictions with class probabilities
        # For binary classification, we work with log-odds
        # For multi-class, we work with one-vs-rest approach
        n_samples = X_train.shape[0]
        
        if self.n_classes_ == 2:
            # Binary classification: use log-odds
            class_counts = np.bincount(y_train)
            initial_pred = np.log(class_counts[1] / class_counts[0])
            current_pred = np.full(n_samples, initial_pred)
            self.initial_pred_ = initial_pred
        else:
            # Multi-class: one-vs-rest with softmax
            current_pred = np.zeros((n_samples, self.n_classes_))
            for k in range(self.n_classes_):
                class_count = np.sum(y_train == self.classes_[k])
                current_pred[:, k] = np.log(class_count / n_samples + 1e-10)
            self.initial_pred_ = np.mean(current_pred, axis=0)
        
        # Boosting iterations
        for stage in range(self.n_estimators):
            learning_rate = self._compute_learning_rate(stage)
            self.learning_rates_.append(learning_rate)
            
            if self.n_classes_ == 2:
                # Compute residuals (negative gradient for log loss)
                probs = 1 / (1 + np.exp(-current_pred))
                residuals = y_train - probs
                
                # Train forest on residuals
                forest = RandomForestClassifier(
                    n_estimators=self.n_trees_per_forest,
                    max_depth=self.max_depth,
                    random_state=self.random_state if self.random_state is None 
                               else self.random_state + stage
                )
                
                # Convert residuals to binary classification problem
                # Use median split to create pseudo-targets
                median_residual = np.median(residuals)
                pseudo_targets = (residuals > median_residual).astype(int)
                
                forest.fit(X_train, pseudo_targets)
                
                # Predict and scale by learning rate
                forest_pred = forest.predict_proba(X_train)[:, 1]
                # Scale to match residual range
                forest_pred = (forest_pred - 0.5) * 2 * np.std(residuals)
                
                current_pred += learning_rate * forest_pred
                
            else:
                # Multi-class: train one forest per class
                forest_list = []
                for k in range(self.n_classes_):
                    # Compute probabilities using softmax
                    exp_pred = np.exp(current_pred - np.max(current_pred, axis=1, keepdims=True))
                    probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
                    
                    # Residuals for class k
                    targets_k = (y_train == self.classes_[k]).astype(float)
                    residuals_k = targets_k - probs[:, k]
                    
                    # Train forest
                    forest = RandomForestClassifier(
                        n_estimators=self.n_trees_per_forest,
                        max_depth=self.max_depth,
                        random_state=self.random_state if self.random_state is None 
                                   else self.random_state + stage * self.n_classes_ + k
                    )
                    
                    median_residual = np.median(residuals_k)
                    pseudo_targets = (residuals_k > median_residual).astype(int)
                    
                    forest.fit(X_train, pseudo_targets)
                    forest_list.append(forest)
                    
                    # Update predictions
                    forest_pred = forest.predict_proba(X_train)[:, 1]
                    forest_pred = (forest_pred - 0.5) * 2 * np.std(residuals_k)
                    current_pred[:, k] += learning_rate * forest_pred
                
                self.forests_.append(forest_list)
                continue
            
            self.forests_.append(forest)
        
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
        
        n_samples = X_test.shape[0]
        
        if self.n_classes_ == 2:
            # Binary classification
            current_pred = np.full(n_samples, self.initial_pred_)
            
            for stage, forest in enumerate(self.forests_):
                learning_rate = self.learning_rates_[stage]
                forest_pred = forest.predict_proba(X_test)[:, 1]
                forest_pred = (forest_pred - 0.5) * 2
                current_pred += learning_rate * forest_pred
            
            # Convert to probabilities
            probs_class1 = 1 / (1 + np.exp(-current_pred))
            proba = np.column_stack([1 - probs_class1, probs_class1])
            
        else:
            # Multi-class
            current_pred = np.tile(self.initial_pred_, (n_samples, 1))
            
            for stage, forest_list in enumerate(self.forests_):
                learning_rate = self.learning_rates_[stage]
                for k, forest in enumerate(forest_list):
                    forest_pred = forest.predict_proba(X_test)[:, 1]
                    forest_pred = (forest_pred - 0.5) * 2
                    current_pred[:, k] += learning_rate * forest_pred
            
            # Softmax
            exp_pred = np.exp(current_pred - np.max(current_pred, axis=1, keepdims=True))
            proba = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
        
        return proba
    
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