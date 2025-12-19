import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class BootstrapMLPEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble of MLPs trained on bootstrap samples with confidence-weighted voting.
    
    Each MLP is trained on a bootstrap sample and votes with weights derived from
    out-of-bag (OOB) error estimates.
    
    Parameters
    ----------
    n_estimators : int, default=10
        Number of MLP estimators in the ensemble.
    hidden_layer_sizes : tuple, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.
    max_iter : int, default=200
        Maximum number of iterations for each MLP.
    random_state : int, default=None
        Random state for reproducibility.
    **mlp_params : dict
        Additional parameters to pass to MLPClassifier.
    """
    
    def __init__(self, n_estimators=10, hidden_layer_sizes=(100,), 
                 max_iter=200, random_state=None, **mlp_params):
        self.n_estimators = n_estimators
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.mlp_params = mlp_params
        
    def fit(self, X_train, y_train):
        """
        Fit the ensemble of MLPs on bootstrap samples.
        
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
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Store estimators and their confidence weights
        self.estimators_ = []
        self.confidence_weights_ = []
        self.oob_indices_ = []
        
        n_samples = X_train.shape[0]
        
        for i in range(self.n_estimators):
            # Create bootstrap sample
            bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
            oob_indices = np.array([idx for idx in range(n_samples) 
                                   if idx not in bootstrap_indices])
            
            X_bootstrap = X_train[bootstrap_indices]
            y_bootstrap = y_train[bootstrap_indices]
            
            # Train MLP on bootstrap sample
            mlp = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                random_state=rng.randint(0, 10000),
                **self.mlp_params
            )
            mlp.fit(X_bootstrap, y_bootstrap)
            
            # Calculate OOB error and confidence weight
            if len(oob_indices) > 0:
                X_oob = X_train[oob_indices]
                y_oob = y_train[oob_indices]
                oob_predictions = mlp.predict(X_oob)
                oob_accuracy = np.mean(oob_predictions == y_oob)
                # Confidence weight based on OOB accuracy
                # Add small epsilon to avoid division by zero
                oob_error = 1.0 - oob_accuracy
                confidence_weight = max(oob_accuracy, 0.01)  # Minimum weight
            else:
                # If no OOB samples, use default weight
                confidence_weight = 0.5
            
            self.estimators_.append(mlp)
            self.confidence_weights_.append(confidence_weight)
            self.oob_indices_.append(oob_indices)
        
        # Normalize confidence weights
        total_weight = sum(self.confidence_weights_)
        if total_weight > 0:
            self.confidence_weights_ = [w / total_weight 
                                       for w in self.confidence_weights_]
        else:
            # Fallback to uniform weights
            self.confidence_weights_ = [1.0 / self.n_estimators] * self.n_estimators
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels using confidence-weighted voting.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
            
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ['estimators_', 'confidence_weights_'])
        X_test = check_array(X_test)
        
        # Get probability predictions from all estimators
        weighted_proba = self.predict_proba(X_test)
        
        # Return class with highest weighted probability
        return self.classes_[np.argmax(weighted_proba, axis=1)]
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities using confidence-weighted voting.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
            
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self, ['estimators_', 'confidence_weights_'])
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        weighted_proba = np.zeros((n_samples, self.n_classes_))
        
        # Aggregate predictions with confidence weights
        for estimator, weight in zip(self.estimators_, self.confidence_weights_):
            proba = estimator.predict_proba(X_test)
            weighted_proba += weight * proba
        
        return weighted_proba
    
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