import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class EnsembleNaiveBayesClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble of Naive Bayes classifiers trained on random feature subsets.
    
    This classifier creates multiple Naive Bayes models, each trained on a
    different random subset of features to reduce variance through ensemble learning.
    
    Parameters
    ----------
    n_estimators : int, default=10
        The number of Naive Bayes models in the ensemble.
    
    max_features : float or int, default=0.7
        The number of features to use for each estimator.
        - If float, represents the fraction of features to use (0.0 to 1.0).
        - If int, represents the absolute number of features to use.
    
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of feature selection for each estimator.
    
    voting : str, default='soft'
        Voting strategy for predictions.
        - 'hard': Uses majority voting.
        - 'soft': Uses average of predicted probabilities.
    
    Attributes
    ----------
    estimators_ : list of GaussianNB
        The collection of fitted Naive Bayes estimators.
    
    feature_subsets_ : list of ndarray
        The feature indices used by each estimator.
    
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    
    n_features_in_ : int
        Number of features seen during fit.
    """
    
    def __init__(self, n_estimators=10, max_features=0.7, random_state=None, voting='soft'):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.voting = voting
    
    def _get_n_features_per_subset(self, n_features):
        """Calculate the number of features to use per subset."""
        if isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        else:
            raise ValueError("max_features must be int or float")
    
    def fit(self, X_train, y_train):
        """
        Fit the ensemble of Naive Bayes classifiers.
        
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
        
        # Store classes and number of features
        self.classes_ = unique_labels(y_train)
        self.n_features_in_ = X_train.shape[1]
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Calculate number of features per subset
        n_features_per_subset = self._get_n_features_per_subset(self.n_features_in_)
        
        # Initialize storage for estimators and feature subsets
        self.estimators_ = []
        self.feature_subsets_ = []
        
        # Train each estimator on a random feature subset
        for i in range(self.n_estimators):
            # Randomly select features
            feature_indices = rng.choice(
                self.n_features_in_,
                size=n_features_per_subset,
                replace=False
            )
            feature_indices = np.sort(feature_indices)
            
            # Create and train a Naive Bayes classifier
            estimator = GaussianNB()
            estimator.fit(X_train[:, feature_indices], y_train)
            
            # Store the estimator and its feature subset
            self.estimators_.append(estimator)
            self.feature_subsets_.append(feature_indices)
        
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
        # Check if fit has been called
        check_is_fitted(self, ['estimators_', 'feature_subsets_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(f"X_test has {X_test.shape[1]} features, "
                           f"but {self.__class__.__name__} was fitted with "
                           f"{self.n_features_in_} features")
        
        # Collect predictions from all estimators
        all_probas = []
        
        for estimator, feature_indices in zip(self.estimators_, self.feature_subsets_):
            proba = estimator.predict_proba(X_test[:, feature_indices])
            all_probas.append(proba)
        
        # Average the probabilities
        avg_proba = np.mean(all_probas, axis=0)
        
        return avg_proba
    
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
        # Check if fit has been called
        check_is_fitted(self, ['estimators_', 'feature_subsets_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(f"X_test has {X_test.shape[1]} features, "
                           f"but {self.__class__.__name__} was fitted with "
                           f"{self.n_features_in_} features")
        
        if self.voting == 'soft':
            # Use probability-based voting
            proba = self.predict_proba(X_test)
            predictions = self.classes_[np.argmax(proba, axis=1)]
        else:
            # Use hard voting (majority vote)
            all_predictions = []
            
            for estimator, feature_indices in zip(self.estimators_, self.feature_subsets_):
                pred = estimator.predict(X_test[:, feature_indices])
                all_predictions.append(pred)
            
            all_predictions = np.array(all_predictions)
            
            # Majority voting
            predictions = []
            for i in range(X_test.shape[0]):
                votes = all_predictions[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                predictions.append(unique[np.argmax(counts)])
            
            predictions = np.array(predictions)
        
        return predictions