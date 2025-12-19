import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.utils import resample
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy import stats


class RadiusBootstrapEnsemble(BaseEstimator, ClassifierMixin):
    """
    Bootstrap Aggregating (Bagging) ensemble of Radius Neighbors Classifiers.
    
    This ensemble trains multiple RadiusNeighborsClassifier models on random
    subsamples of the training data with varying radius parameters, then
    combines their predictions through majority voting.
    
    Parameters
    ----------
    n_estimators : int, default=10
        The number of base estimators in the ensemble.
    
    radius_min : float, default=0.5
        Minimum radius for the radius neighbors classifier.
    
    radius_max : float, default=2.0
        Maximum radius for the radius neighbors classifier.
    
    max_samples : float or int, default=1.0
        The number of samples to draw for training each base estimator.
        - If float, should be between 0.0 and 1.0 and represents the fraction
          of the dataset to sample.
        - If int, represents the absolute number of samples.
    
    outlier_label : int or None, default=None
        Label for outlier samples (samples with no neighbors within radius).
        If None, uses the most common class in training data.
    
    weights : str, default='uniform'
        Weight function used in prediction. Possible values:
        - 'uniform' : uniform weights (all points in each neighborhood are weighted equally)
        - 'distance' : weight points by the inverse of their distance
    
    random_state : int or None, default=None
        Controls the random resampling of the original dataset.
    
    Attributes
    ----------
    estimators_ : list of RadiusNeighborsClassifier
        The collection of fitted base estimators.
    
    radii_ : ndarray of shape (n_estimators,)
        The radius parameter used for each estimator.
    
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    """
    
    def __init__(
        self,
        n_estimators=10,
        radius_min=0.5,
        radius_max=2.0,
        max_samples=1.0,
        outlier_label=None,
        weights='uniform',
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.max_samples = max_samples
        self.outlier_label = outlier_label
        self.weights = weights
        self.random_state = random_state
    
    def fit(self, X_train, y_train):
        """
        Fit the ensemble of radius classifiers.
        
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
        
        # Store classes
        self.classes_ = np.unique(y_train)
        
        # Set outlier label if not provided
        if self.outlier_label is None:
            # Use the most common class as default outlier label
            mode_result = stats.mode(y_train, keepdims=True)
            self.outlier_label_ = mode_result[0][0]
        else:
            self.outlier_label_ = self.outlier_label
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Generate varying radius parameters
        if self.n_estimators == 1:
            self.radii_ = np.array([(self.radius_min + self.radius_max) / 2])
        else:
            self.radii_ = np.linspace(self.radius_min, self.radius_max, self.n_estimators)
        
        # Initialize estimators list
        self.estimators_ = []
        
        # Train each estimator on a bootstrap sample
        for i in range(self.n_estimators):
            # Create bootstrap sample
            X_sample, y_sample = resample(
                X_train,
                y_train,
                n_samples=self._get_n_samples(X_train.shape[0]),
                random_state=rng.randint(0, np.iinfo(np.int32).max),
                replace=True
            )
            
            # Create and fit radius classifier
            # Note: outlier_label should be a scalar or 'most_frequent'
            # We'll use 'most_frequent' to let sklearn handle it automatically
            estimator = RadiusNeighborsClassifier(
                radius=self.radii_[i],
                weights=self.weights,
                outlier_label='most_frequent'  # Let sklearn handle outliers
            )
            estimator.fit(X_sample, y_sample)
            
            self.estimators_.append(estimator)
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test using majority voting.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fitted
        check_is_fitted(self, ['estimators_', 'classes_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Collect predictions from all estimators
        predictions = np.array([
            estimator.predict(X_test)
            for estimator in self.estimators_
        ])
        
        # Perform majority voting
        y_pred = np.array([
            self._majority_vote(predictions[:, i])
            for i in range(X_test.shape[0])
        ])
        
        return y_pred
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        # Check if fitted
        check_is_fitted(self, ['estimators_', 'classes_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Collect predictions from all estimators
        predictions = np.array([
            estimator.predict(X_test)
            for estimator in self.estimators_
        ])
        
        # Calculate probabilities based on voting
        n_samples = X_test.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            for j, cls in enumerate(self.classes_):
                proba[i, j] = np.sum(predictions[:, i] == cls) / self.n_estimators
        
        return proba
    
    def _get_n_samples(self, n_total):
        """Calculate the number of samples to draw for bootstrap."""
        if isinstance(self.max_samples, float):
            return int(self.max_samples * n_total)
        else:
            return min(self.max_samples, n_total)
    
    def _majority_vote(self, predictions):
        """Perform majority voting on predictions."""
        values, counts = np.unique(predictions, return_counts=True)
        return values[np.argmax(counts)]