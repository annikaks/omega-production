import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class ResidualMLPEnsemble(BaseEstimator, ClassifierMixin):
    """
    Sequential MLP ensemble where each network learns residuals of previous predictions.
    Uses exponentially weighted combination for final predictions.
    """
    
    def __init__(self, n_estimators=5, hidden_layer_sizes=(100,), 
                 activation='relu', solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate='constant',
                 learning_rate_init=0.001, max_iter=200, 
                 random_state=None, weight_decay=0.5):
        """
        Parameters
        ----------
        n_estimators : int, default=5
            Number of sequential MLP models to train
        hidden_layer_sizes : tuple, default=(100,)
            Hidden layer sizes for each MLP
        activation : str, default='relu'
            Activation function
        solver : str, default='adam'
            Optimizer
        alpha : float, default=0.0001
            L2 regularization parameter
        batch_size : int or 'auto', default='auto'
            Batch size for training
        learning_rate : str, default='constant'
            Learning rate schedule
        learning_rate_init : float, default=0.001
            Initial learning rate
        max_iter : int, default=200
            Maximum iterations for each MLP
        random_state : int, default=None
            Random seed
        weight_decay : float, default=0.5
            Exponential decay factor for ensemble weights (0 < weight_decay <= 1)
        """
        self.n_estimators = n_estimators
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.weight_decay = weight_decay
        
    def fit(self, X_train, y_train):
        """
        Fit the sequential residual MLP ensemble.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        
        # Initialize label binarizer for multi-class
        self.label_binarizer_ = LabelBinarizer()
        y_binary = self.label_binarizer_.fit_transform(y_train)
        if y_binary.shape[1] == 1:
            y_binary = np.hstack([1 - y_binary, y_binary])
        
        # Initialize storage
        self.estimators_ = []
        self.ensemble_weights_ = []
        
        # Current residuals (start with actual labels)
        current_target = y_binary.copy()
        
        # Ensemble predictions accumulator
        ensemble_pred = np.zeros_like(y_binary, dtype=float)
        
        # Train sequential models
        for i in range(self.n_estimators):
            # Create MLP for this stage
            random_state = None if self.random_state is None else self.random_state + i
            
            mlp = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                random_state=random_state,
                warm_start=False
            )
            
            # For residual learning, we need to predict probabilities
            # Create temporary labels from residuals
            residual_labels = np.argmax(current_target, axis=1)
            
            # Fit MLP on current residuals
            mlp.fit(X_train, residual_labels)
            
            # Get predictions as probabilities
            pred_proba = mlp.predict_proba(X_train)
            
            # Ensure prediction shape matches target shape
            if pred_proba.shape[1] != y_binary.shape[1]:
                # Handle binary case
                pred_proba_full = np.zeros((pred_proba.shape[0], y_binary.shape[1]))
                for j, cls in enumerate(mlp.classes_):
                    if cls < y_binary.shape[1]:
                        pred_proba_full[:, cls] = pred_proba[:, j]
                pred_proba = pred_proba_full
            
            # Calculate exponential weight for this estimator
            weight = self.weight_decay ** i
            self.ensemble_weights_.append(weight)
            
            # Update ensemble predictions
            ensemble_pred += weight * pred_proba
            
            # Calculate new residuals (difference between target and ensemble prediction)
            normalized_ensemble = ensemble_pred / np.sum(self.ensemble_weights_)
            current_target = y_binary - normalized_ensemble
            
            # Store estimator
            self.estimators_.append(mlp)
        
        # Normalize weights
        self.ensemble_weights_ = np.array(self.ensemble_weights_)
        self.ensemble_weights_ /= np.sum(self.ensemble_weights_)
        
        return self
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        # Initialize predictions
        n_samples = X_test.shape[0]
        ensemble_pred = np.zeros((n_samples, self.n_classes_))
        
        # Accumulate weighted predictions
        for estimator, weight in zip(self.estimators_, self.ensemble_weights_):
            pred_proba = estimator.predict_proba(X_test)
            
            # Handle shape mismatch
            if pred_proba.shape[1] != self.n_classes_:
                pred_proba_full = np.zeros((n_samples, self.n_classes_))
                for j, cls in enumerate(estimator.classes_):
                    if cls < self.n_classes_:
                        pred_proba_full[:, cls] = pred_proba[:, j]
                pred_proba = pred_proba_full
            
            ensemble_pred += weight * pred_proba
        
        # Normalize to ensure valid probabilities
        ensemble_pred = np.clip(ensemble_pred, 0, 1)
        row_sums = ensemble_pred.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        ensemble_pred = ensemble_pred / row_sums
        
        return ensemble_pred
    
    def predict(self, X_test):
        """
        Predict class labels for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]