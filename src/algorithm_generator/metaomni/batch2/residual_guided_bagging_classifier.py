import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets


class ResidualGuidedBaggingClassifier(BaseEstimator, ClassifierMixin):
    """
    Residual-Guided Bagging Classifier where each successive tree is trained
    on a bootstrap sample weighted by the previous ensemble's prediction uncertainty.
    
    Parameters
    ----------
    base_estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a DecisionTreeClassifier.
    
    n_estimators : int, default=10
        The number of base estimators in the ensemble.
    
    max_samples : float or int, default=1.0
        The number of samples to draw from X to train each base estimator.
        - If float, then draw max_samples * n_samples samples.
        - If int, then draw max_samples samples.
    
    random_state : int, default=None
        Controls the random resampling of the original dataset.
    
    uncertainty_power : float, default=1.0
        Power to raise the uncertainty weights to. Higher values give more
        weight to uncertain predictions.
    """
    
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0,
                 random_state=None, uncertainty_power=1.0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.uncertainty_power = uncertainty_power
    
    def fit(self, X_train, y_train):
        """
        Fit the residual-guided bagging classifier.
        
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
        check_classification_targets(y_train)
        
        self.classes_ = np.unique(y_train)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_train.shape[1]
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Initialize base estimator
        if self.base_estimator is None:
            base_est = DecisionTreeClassifier(random_state=self.random_state)
        else:
            base_est = self.base_estimator
        
        # Calculate max samples
        n_samples = X_train.shape[0]
        if isinstance(self.max_samples, float):
            n_samples_bootstrap = int(self.max_samples * n_samples)
        else:
            n_samples_bootstrap = self.max_samples
        n_samples_bootstrap = min(n_samples_bootstrap, n_samples)
        
        # Initialize estimators list
        self.estimators_ = []
        
        # Initialize uniform weights for first iteration
        sample_weights = np.ones(n_samples)
        
        for i in range(self.n_estimators):
            # Normalize weights
            sample_weights = sample_weights / sample_weights.sum()
            
            # Bootstrap sampling with weights
            indices = rng.choice(
                n_samples,
                size=n_samples_bootstrap,
                replace=True,
                p=sample_weights
            )
            
            # Get bootstrap sample
            X_bootstrap = X_train[indices]
            y_bootstrap = y_train[indices]
            
            # Clone and fit estimator
            estimator = clone(base_est)
            estimator.fit(X_bootstrap, y_bootstrap)
            self.estimators_.append(estimator)
            
            # Calculate uncertainty for next iteration
            if i < self.n_estimators - 1:
                # Get predictions from current ensemble
                ensemble_proba = self._predict_proba_ensemble(X_train, i + 1)
                
                # Calculate uncertainty (entropy-based)
                # Higher entropy = higher uncertainty
                epsilon = 1e-10
                entropy = -np.sum(
                    ensemble_proba * np.log(ensemble_proba + epsilon),
                    axis=1
                )
                
                # Normalize entropy to [0, 1]
                max_entropy = np.log(self.n_classes_)
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else entropy
                
                # Calculate prediction correctness
                ensemble_pred = self.classes_[np.argmax(ensemble_proba, axis=1)]
                is_correct = (ensemble_pred == y_train).astype(float)
                
                # Combine uncertainty and incorrectness
                # Higher weight for uncertain and incorrect predictions
                uncertainty = normalized_entropy
                incorrectness = 1 - is_correct
                
                # Weight is combination of uncertainty and incorrectness
                sample_weights = (uncertainty + incorrectness) / 2
                
                # Apply power transformation
                sample_weights = np.power(sample_weights + 1e-10, self.uncertainty_power)
                
                # Add small constant to avoid zero weights
                sample_weights = sample_weights + 0.1
        
        return self
    
    def _predict_proba_ensemble(self, X, n_estimators):
        """
        Predict class probabilities using the first n_estimators.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        n_estimators : int
            Number of estimators to use.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        probas = []
        for estimator in self.estimators_[:n_estimators]:
            if hasattr(estimator, 'predict_proba'):
                proba = estimator.predict_proba(X)
            else:
                # For estimators without predict_proba, use one-hot encoding
                pred = estimator.predict(X)
                proba = np.zeros((X.shape[0], self.n_classes_))
                for idx, cls in enumerate(self.classes_):
                    proba[pred == cls, idx] = 1.0
            probas.append(proba)
        
        # Average probabilities
        return np.mean(probas, axis=0)
    
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
        
        return self._predict_proba_ensemble(X_test, len(self.estimators_))
    
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