import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class DynamicLambdaClassifier(BaseEstimator, ClassifierMixin):
    """
    A logistic regression classifier with dynamic lambda scheduling that decreases
    regularization strength as training progresses.
    
    Parameters
    ----------
    lambda_init : float, default=1.0
        Initial regularization strength (L2 penalty).
    lambda_final : float, default=0.01
        Final regularization strength.
    schedule : str, default='exponential'
        Type of schedule: 'exponential', 'linear', or 'cosine'.
    max_iter : int, default=1000
        Maximum number of iterations for optimization.
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    tol : float, default=1e-4
        Tolerance for stopping criterion.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, lambda_init=1.0, lambda_final=0.01, schedule='exponential',
                 max_iter=1000, learning_rate=0.01, tol=1e-4, random_state=None):
        self.lambda_init = lambda_init
        self.lambda_final = lambda_final
        self.schedule = schedule
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.random_state = random_state
    
    def _get_lambda(self, iteration):
        """Calculate regularization strength at given iteration."""
        progress = iteration / self.max_iter
        
        if self.schedule == 'exponential':
            # Exponential decay
            lambda_t = self.lambda_init * np.exp(
                progress * np.log(self.lambda_final / self.lambda_init)
            )
        elif self.schedule == 'linear':
            # Linear decay
            lambda_t = self.lambda_init + progress * (self.lambda_final - self.lambda_init)
        elif self.schedule == 'cosine':
            # Cosine annealing
            lambda_t = self.lambda_final + 0.5 * (self.lambda_init - self.lambda_final) * (
                1 + np.cos(np.pi * progress)
            )
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
        
        return lambda_t
    
    def _sigmoid(self, z):
        """Compute sigmoid function with numerical stability."""
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )
    
    def _softmax(self, z):
        """Compute softmax function with numerical stability."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _compute_loss(self, X, y, weights, lambda_t):
        """Compute cross-entropy loss with L2 regularization."""
        n_samples = X.shape[0]
        
        if self.n_classes_ == 2:
            # Binary classification
            z = X @ weights
            predictions = self._sigmoid(z)
            loss = -np.mean(y * np.log(predictions + 1e-15) + 
                           (1 - y) * np.log(1 - predictions + 1e-15))
        else:
            # Multi-class classification
            z = X @ weights
            predictions = self._softmax(z)
            loss = -np.mean(np.sum(y * np.log(predictions + 1e-15), axis=1))
        
        # Add L2 regularization (excluding bias term)
        reg_loss = 0.5 * lambda_t * np.sum(weights[1:] ** 2)
        
        return loss + reg_loss
    
    def _compute_gradient(self, X, y, weights, lambda_t):
        """Compute gradient of loss function."""
        n_samples = X.shape[0]
        
        if self.n_classes_ == 2:
            # Binary classification
            z = X @ weights
            predictions = self._sigmoid(z)
            gradient = X.T @ (predictions - y) / n_samples
        else:
            # Multi-class classification
            z = X @ weights
            predictions = self._softmax(z)
            gradient = X.T @ (predictions - y) / n_samples
        
        # Add L2 regularization gradient (excluding bias term)
        gradient[1:] += lambda_t * weights[1:]
        
        return gradient
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier with dynamic lambda scheduling.
        
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
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y_train)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        
        # Add bias term
        X_bias = np.c_[np.ones(X_train.shape[0]), X_train]
        n_features = X_bias.shape[1]
        
        # Initialize weights
        rng = np.random.RandomState(self.random_state)
        
        if self.n_classes_ == 2:
            # Binary classification
            self.weights_ = rng.randn(n_features) * 0.01
            y_train_encoded = y_encoded.astype(float)
        else:
            # Multi-class classification (one-vs-rest)
            self.weights_ = rng.randn(n_features, self.n_classes_) * 0.01
            y_train_encoded = np.eye(self.n_classes_)[y_encoded]
        
        # Training loop with dynamic lambda
        self.loss_history_ = []
        self.lambda_history_ = []
        prev_loss = float('inf')
        
        for iteration in range(self.max_iter):
            # Get current lambda value
            lambda_t = self._get_lambda(iteration)
            self.lambda_history_.append(lambda_t)
            
            # Compute gradient and update weights
            gradient = self._compute_gradient(X_bias, y_train_encoded, 
                                             self.weights_, lambda_t)
            self.weights_ -= self.learning_rate * gradient
            
            # Compute loss
            loss = self._compute_loss(X_bias, y_train_encoded, 
                                     self.weights_, lambda_t)
            self.loss_history_.append(loss)
            
            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                break
            
            prev_loss = loss
        
        self.n_iter_ = iteration + 1
        
        return self
    
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
        
        # Add bias term
        X_bias = np.c_[np.ones(X_test.shape[0]), X_test]
        
        if self.n_classes_ == 2:
            # Binary classification
            z = X_bias @ self.weights_
            proba_pos = self._sigmoid(z)
            return np.c_[1 - proba_pos, proba_pos]
        else:
            # Multi-class classification
            z = X_bias @ self.weights_
            return self._softmax(z)
    
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
        y_pred_encoded = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(y_pred_encoded)
    
    def get_regularization_schedule(self):
        """
        Get the regularization schedule used during training.
        
        Returns
        -------
        lambda_history : array
            Regularization strength at each iteration.
        """
        check_is_fitted(self)
        return np.array(self.lambda_history_)