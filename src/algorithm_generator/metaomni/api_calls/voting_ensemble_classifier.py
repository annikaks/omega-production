from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import numpy as np


class VotingEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier combining Logistic Regression, HistGradientBoosting, 
    and Random Forest using soft voting.
    
    Parameters
    ----------
    lr_C : float, default=1.0
        Inverse of regularization strength for Logistic Regression.
    
    lr_max_iter : int, default=1000
        Maximum number of iterations for Logistic Regression.
    
    hgb_max_iter : int, default=100
        Maximum number of iterations for HistGradientBoosting.
    
    hgb_learning_rate : float, default=0.1
        Learning rate for HistGradientBoosting.
    
    rf_n_estimators : int, default=100
        Number of trees in Random Forest.
    
    rf_max_depth : int or None, default=None
        Maximum depth of Random Forest trees.
    
    voting : str, default='soft'
        Voting strategy: 'hard' for majority voting, 'soft' for weighted average.
    
    weights : array-like of shape (3,), default=None
        Weights for each classifier. If None, uniform weights are used.
    
    random_state : int or None, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, lr_C=1.0, lr_max_iter=1000, 
                 hgb_max_iter=100, hgb_learning_rate=0.1,
                 rf_n_estimators=100, rf_max_depth=None,
                 voting='soft', weights=None, random_state=None):
        self.lr_C = lr_C
        self.lr_max_iter = lr_max_iter
        self.hgb_max_iter = hgb_max_iter
        self.hgb_learning_rate = hgb_learning_rate
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.voting = voting
        self.weights = weights
        self.random_state = random_state
    
    def fit(self, X, y):
        """
        Fit the ensemble classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Initialize classifiers
        self.lr_ = LogisticRegression(
            C=self.lr_C,
            max_iter=self.lr_max_iter,
            random_state=self.random_state
        )
        
        self.hgb_ = HistGradientBoostingClassifier(
            max_iter=self.hgb_max_iter,
            learning_rate=self.hgb_learning_rate,
            random_state=self.random_state
        )
        
        self.rf_ = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            random_state=self.random_state
        )
        
        # Fit all classifiers
        self.lr_.fit(X, y)
        self.hgb_.fit(X, y)
        self.rf_.fit(X, y)
        
        # Set weights
        if self.weights is None:
            self.weights_ = np.ones(3) / 3
        else:
            self.weights_ = np.array(self.weights)
            self.weights_ = self.weights_ / np.sum(self.weights_)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # Get predictions from each classifier
        lr_proba = self.lr_.predict_proba(X)
        hgb_proba = self.hgb_.predict_proba(X)
        rf_proba = self.rf_.predict_proba(X)
        
        # Weighted average
        avg_proba = (self.weights_[0] * lr_proba + 
                     self.weights_[1] * hgb_proba + 
                     self.weights_[2] * rf_proba)
        
        return avg_proba
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        if self.voting == 'soft':
            # Use probability-based voting
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
        else:
            # Hard voting
            lr_pred = self.lr_.predict(X)
            hgb_pred = self.hgb_.predict(X)
            rf_pred = self.rf_.predict(X)
            
            # Stack predictions
            predictions = np.column_stack([lr_pred, hgb_pred, rf_pred])
            
            # Majority vote with weights
            y_pred = np.empty(X.shape[0], dtype=self.classes_.dtype)
            for i in range(X.shape[0]):
                votes = {}
                for j, pred in enumerate(predictions[i]):
                    votes[pred] = votes.get(pred, 0) + self.weights_[j]
                y_pred[i] = max(votes, key=votes.get)
            
            return y_pred