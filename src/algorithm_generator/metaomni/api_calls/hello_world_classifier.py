from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class HelloWorldClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple classifier that predicts 'Hello World' or 'Not Hello World'.
    
    This classifier checks if input text contains variations of 'hello world'
    and classifies accordingly.
    
    Parameters
    ----------
    case_sensitive : bool, default=False
        Whether to perform case-sensitive matching.
    
    partial_match : bool, default=True
        Whether to allow partial matches (e.g., just 'hello' or just 'world').
    """
    
    def __init__(self, case_sensitive=False, partial_match=True):
        self.case_sensitive = case_sensitive
        self.partial_match = partial_match
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier (no-op for this simple rule-based classifier).
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples,)
            Training data (text strings).
        
        y_train : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        # Store classes seen during fit
        self.classes_ = np.unique(y_train)
        
        # For a rule-based classifier, we don't need to learn anything
        # but we validate the input
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length")
        
        self.n_features_in_ = 1  # We treat each string as a single feature
        
        return self
    
    def predict(self, X_test):
        """
        Predict whether inputs are 'Hello World' or not.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples,)
            Test data (text strings).
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        X_test = np.asarray(X_test)
        predictions = []
        
        for text in X_test:
            text_str = str(text)
            
            if not self.case_sensitive:
                text_str = text_str.lower()
                hello_check = 'hello'
                world_check = 'world'
            else:
                hello_check = 'hello'
                world_check = 'world'
            
            if self.partial_match:
                # Match if either 'hello' or 'world' is present
                is_hello_world = hello_check in text_str or world_check in text_str
            else:
                # Match only if both 'hello' and 'world' are present
                is_hello_world = hello_check in text_str and world_check in text_str
            
            predictions.append(1 if is_hello_world else 0)
        
        return np.array(predictions)
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples,)
            Test data (text strings).
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Probability estimates.
        """
        predictions = self.predict(X_test)
        n_samples = len(predictions)
        n_classes = len(self.classes_) if hasattr(self, 'classes_') else 2
        
        proba = np.zeros((n_samples, n_classes))
        for i, pred in enumerate(predictions):
            proba[i, pred] = 1.0
        
        return proba