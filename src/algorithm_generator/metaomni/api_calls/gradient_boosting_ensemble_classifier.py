import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import mode


class GradientBoostingEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble of Gradient Boosting Machines classifier.
    
    This classifier creates an ensemble of multiple Gradient Boosting classifiers
    with different hyperparameters or random states to improve prediction robustness
    and generalization.
    
    Parameters
    ----------
    n_estimators_per_model : int, default=100
        The number of boosting stages to perform for each GBM in the ensemble.
        
    n_models : int, default=5
        The number of GBM models in the ensemble.
        
    learning_rate : float or list of float, default=0.1
        Learning rate shrinks the contribution of each tree. If a list is provided,
        each model will use a different learning rate.
        
    max_depth : int or list of int, default=3
        Maximum depth of the individual regression estimators. If a list is provided,
        each model will use a different max_depth.
        
    subsample : float or list of float, default=1.0
        The fraction of samples to be used for fitting the individual base learners.
        If a list is provided, each model will use a different subsample rate.
        
    voting : str, default='soft'
        If 'hard', uses predicted class labels for majority rule voting.
        If 'soft', predicts the class label based on the argmax of the sums of
        the predicted probabilities.
        
    random_state : int or None, default=None
        Controls the random seed given to each GBM model.
        
    Attributes
    ----------
    models_ : list
        The collection of fitted GBM models.
        
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
        
    n_classes_ : int
        The number of classes.
    """
    
    def __init__(self, n_estimators_per_model=100, n_models=5, learning_rate=0.1,
                 max_depth=3, subsample=1.0, voting='soft', random_state=None):
        self.n_estimators_per_model = n_estimators_per_model
        self.n_models = n_models
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.voting = voting
        self.random_state = random_state
    
    def _get_param_list(self, param, n_models):
        """Convert parameter to list if it's a single value."""
        if isinstance(param, (list, tuple, np.ndarray)):
            if len(param) != n_models:
                raise ValueError(f"Length of parameter list must match n_models ({n_models})")
            return list(param)
        else:
            return [param] * n_models
    
    def fit(self, X, y):
        """
        Fit the ensemble of Gradient Boosting classifiers.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
            
        y : array-like of shape (n_samples,)
            The target values (class labels).
            
        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        # Convert parameters to lists
        learning_rates = self._get_param_list(self.learning_rate, self.n_models)
        max_depths = self._get_param_list(self.max_depth, self.n_models)
        subsamples = self._get_param_list(self.subsample, self.n_models)
        
        # Initialize and fit models
        self.models_ = []
        
        for i in range(self.n_models):
            # Set random state for reproducibility
            if self.random_state is not None:
                model_random_state = self.random_state + i
            else:
                model_random_state = None
            
            # Create GBM model with specific parameters
            model = GradientBoostingClassifier(
                n_estimators=self.n_estimators_per_model,
                learning_rate=learning_rates[i],
                max_depth=max_depths[i],
                subsample=subsamples[i],
                random_state=model_random_state
            )
            
            # Fit the model
            model.fit(X, y)
            self.models_.append(model)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        # Check is fit had been called
        check_is_fitted(self, ['models_', 'classes_'])
        
        # Input validation
        X = check_array(X)
        
        # Aggregate predictions from all models
        all_probas = np.array([model.predict_proba(X) for model in self.models_])
        
        # Average probabilities across all models
        avg_proba = np.mean(all_probas, axis=0)
        
        return avg_proba
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
            
        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted class labels.
        """
        # Check is fit had been called
        check_is_fitted(self, ['models_', 'classes_'])
        
        # Input validation
        X = check_array(X)
        
        if self.voting == 'soft':
            # Use averaged probabilities
            proba = self.predict_proba(X)
            predictions = self.classes_[np.argmax(proba, axis=1)]
        else:
            # Hard voting: majority vote
            all_predictions = np.array([model.predict(X) for model in self.models_])
            # Use mode for majority voting
            predictions, _ = mode(all_predictions, axis=0, keepdims=False)
        
        return predictions
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
            
        y : array-like of shape (n_samples,)
            True labels for X.
            
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))