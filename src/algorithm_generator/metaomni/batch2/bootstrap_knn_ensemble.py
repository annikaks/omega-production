import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class BootstrapKNNEnsemble(BaseEstimator, ClassifierMixin):
    """
    An ensemble of KNN classifiers trained on bootstrapped samples with randomized k values.
    Predictions are aggregated through weighted voting based on out-of-bag accuracy.
    
    Parameters
    ----------
    n_estimators : int, default=10
        The number of KNN classifiers in the ensemble.
    k_range : tuple, default=(3, 15)
        The range (min, max) for randomly selecting k values for each KNN classifier.
    bootstrap_ratio : float, default=1.0
        The ratio of samples to draw for each bootstrap sample (with replacement).
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_estimators=10, k_range=(3, 15), bootstrap_ratio=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.k_range = k_range
        self.bootstrap_ratio = bootstrap_ratio
        self.random_state = random_state
        
    def fit(self, X_train, y_train):
        """
        Fit the ensemble of KNN classifiers on bootstrapped samples.
        
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
        self.n_features_in_ = X_train.shape[1]
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Store classifiers, their k values, and weights
        self.estimators_ = []
        self.k_values_ = []
        self.weights_ = []
        self.bootstrap_indices_ = []
        
        n_samples = X_train.shape[0]
        bootstrap_size = int(n_samples * self.bootstrap_ratio)
        
        for i in range(self.n_estimators):
            # Generate bootstrap sample
            bootstrap_idx = rng.choice(n_samples, size=bootstrap_size, replace=True)
            oob_idx = np.array([idx for idx in range(n_samples) if idx not in bootstrap_idx])
            
            X_bootstrap = X_train[bootstrap_idx]
            y_bootstrap = y_train[bootstrap_idx]
            
            # Randomly select k value
            k = rng.randint(self.k_range[0], self.k_range[1] + 1)
            
            # Train KNN classifier
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_bootstrap, y_bootstrap)
            
            # Calculate weight based on OOB accuracy
            weight = 1.0  # Default weight
            if len(oob_idx) > 0:
                X_oob = X_train[oob_idx]
                y_oob = y_train[oob_idx]
                oob_score = knn.score(X_oob, y_oob)
                weight = max(oob_score, 0.01)  # Avoid zero weights
            
            self.estimators_.append(knn)
            self.k_values_.append(k)
            self.weights_.append(weight)
            self.bootstrap_indices_.append(bootstrap_idx)
        
        # Normalize weights
        total_weight = sum(self.weights_)
        self.weights_ = [w / total_weight for w in self.weights_]
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test using weighted voting.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fitted
        check_is_fitted(self, ['estimators_', 'weights_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        n_classes = len(self.classes_)
        
        # Accumulate weighted votes
        weighted_votes = np.zeros((n_samples, n_classes))
        
        for estimator, weight in zip(self.estimators_, self.weights_):
            predictions = estimator.predict(X_test)
            
            # Convert predictions to class indices and accumulate weighted votes
            for i, pred in enumerate(predictions):
                class_idx = np.where(self.classes_ == pred)[0][0]
                weighted_votes[i, class_idx] += weight
        
        # Return class with highest weighted vote
        y_pred = self.classes_[np.argmax(weighted_votes, axis=1)]
        
        return y_pred
    
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
        # Check if fitted
        check_is_fitted(self, ['estimators_', 'weights_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        n_classes = len(self.classes_)
        
        # Accumulate weighted probabilities
        weighted_proba = np.zeros((n_samples, n_classes))
        
        for estimator, weight in zip(self.estimators_, self.weights_):
            proba = estimator.predict_proba(X_test)
            weighted_proba += weight * proba
        
        return weighted_proba