import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


class BiasVarianceMultinomialClassifier(BaseEstimator, ClassifierMixin):
    """
    Decompose multinomial probabilities into bias term plus variance-reducing 
    residual corrections learned from validation errors.
    
    This classifier learns a base model (bias term) and then learns residual
    corrections from validation errors to reduce variance in predictions.
    
    Parameters
    ----------
    base_estimator : estimator object, default=None
        The base classifier to use. If None, uses LogisticRegression.
    
    residual_estimator : estimator object, default=None
        The estimator for learning residual corrections. If None, uses LogisticRegression.
    
    validation_size : float, default=0.2
        Proportion of training data to use for learning residual corrections.
    
    residual_weight : float, default=0.5
        Weight for combining base predictions with residual corrections.
        Final prediction = (1 - residual_weight) * base + residual_weight * residual
    
    random_state : int, default=None
        Random state for train/validation split.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    
    base_model_ : estimator
        The fitted base model (bias term).
    
    residual_model_ : estimator
        The fitted residual correction model.
    """
    
    def __init__(self, base_estimator=None, residual_estimator=None,
                 validation_size=0.2, residual_weight=0.5, random_state=None):
        self.base_estimator = base_estimator
        self.residual_estimator = residual_estimator
        self.validation_size = validation_size
        self.residual_weight = residual_weight
        self.random_state = random_state
    
    def fit(self, X_train, y_train):
        """
        Fit the bias-variance decomposed classifier.
        
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
        
        # Initialize base estimator
        if self.base_estimator is None:
            self.base_model_ = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                random_state=self.random_state
            )
        else:
            self.base_model_ = self.base_estimator
        
        # Initialize residual estimator
        if self.residual_estimator is None:
            self.residual_model_ = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                random_state=self.random_state
            )
        else:
            self.residual_model_ = self.residual_estimator
        
        # Split training data into train and validation
        if self.validation_size > 0:
            X_base, X_val, y_base, y_val = train_test_split(
                X_train, y_train,
                test_size=self.validation_size,
                random_state=self.random_state,
                stratify=y_train
            )
        else:
            X_base, X_val = X_train, X_train
            y_base, y_val = y_train, y_train
        
        # Fit base model (bias term)
        self.base_model_.fit(X_base, y_base)
        
        # Get base predictions on validation set
        base_proba_val = self.base_model_.predict_proba(X_val)
        
        # Compute residuals (errors) on validation set
        # Create one-hot encoded target
        y_val_onehot = np.zeros((len(y_val), self.n_classes_))
        for i, cls in enumerate(self.classes_):
            y_val_onehot[y_val == cls, i] = 1.0
        
        # Residuals are the difference between true probabilities and predicted
        residuals = y_val_onehot - base_proba_val
        
        # Create augmented features for residual learning
        # Concatenate original features with base predictions
        X_val_augmented = np.hstack([X_val, base_proba_val])
        
        # Learn residual corrections by predicting the errors
        # We'll use the sign of residuals to create targets for classification
        # Use the class with maximum positive residual as target
        residual_targets = np.argmax(residuals, axis=1)
        residual_targets = self.classes_[residual_targets]
        
        # Fit residual model
        self.residual_model_.fit(X_val_augmented, residual_targets)
        
        # Store residual statistics for normalization
        self.residual_mean_ = np.mean(np.abs(residuals), axis=0)
        self.residual_std_ = np.std(residuals, axis=0) + 1e-8
        
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
            Class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        # Get base predictions (bias term)
        base_proba = self.base_model_.predict_proba(X_test)
        
        # Get residual corrections
        X_test_augmented = np.hstack([X_test, base_proba])
        residual_proba = self.residual_model_.predict_proba(X_test_augmented)
        
        # Compute residual correction as deviation from uniform
        uniform_proba = np.ones(self.n_classes_) / self.n_classes_
        residual_correction = residual_proba - uniform_proba
        
        # Combine base predictions with residual corrections
        combined_proba = (
            (1 - self.residual_weight) * base_proba +
            self.residual_weight * (base_proba + residual_correction)
        )
        
        # Ensure probabilities are valid (non-negative and sum to 1)
        combined_proba = np.maximum(combined_proba, 0)
        combined_proba = combined_proba / combined_proba.sum(axis=1, keepdims=True)
        
        return combined_proba
    
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
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        y : array-like of shape (n_samples,)
            True labels.
        
        Returns
        -------
        score : float
            Mean accuracy.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))