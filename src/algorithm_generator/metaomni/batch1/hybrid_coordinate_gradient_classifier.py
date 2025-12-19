import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class HybridCoordinateGradientClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid optimizer combining coordinate-wise closed-form updates with gradient descent.
    
    This classifier alternates between:
    1. Coordinate-wise closed-form updates for weights (when possible)
    2. Gradient descent steps for bias and regularization
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent steps.
    n_iterations : int, default=100
        Number of optimization iterations.
    n_coord_updates : int, default=5
        Number of coordinate updates per iteration.
    n_grad_steps : int, default=3
        Number of gradient steps per iteration.
    reg_lambda : float, default=0.01
        L2 regularization parameter.
    tol : float, default=1e-4
        Tolerance for convergence.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=100, 
                 n_coord_updates=5, n_grad_steps=3, reg_lambda=0.01,
                 tol=1e-4, random_state=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.n_coord_updates = n_coord_updates
        self.n_grad_steps = n_grad_steps
        self.reg_lambda = reg_lambda
        self.tol = tol
        self.random_state = random_state
    
    def _sigmoid(self, z):
        """Sigmoid activation function with numerical stability."""
        return np.where(z >= 0, 
                       1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))
    
    def _softmax(self, z):
        """Softmax function with numerical stability."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _compute_loss(self, X, y_encoded):
        """Compute cross-entropy loss with L2 regularization."""
        n_samples = X.shape[0]
        logits = X @ self.coef_.T + self.intercept_
        
        if self.n_classes_ == 2:
            probs = self._sigmoid(logits.ravel())
            loss = -np.mean(y_encoded.ravel() * np.log(probs + 1e-15) + 
                           (1 - y_encoded.ravel()) * np.log(1 - probs + 1e-15))
        else:
            probs = self._softmax(logits)
            loss = -np.mean(np.sum(y_encoded * np.log(probs + 1e-15), axis=1))
        
        # Add L2 regularization
        loss += 0.5 * self.reg_lambda * np.sum(self.coef_ ** 2)
        return loss
    
    def _coordinate_update(self, X, y_encoded, coord_idx):
        """
        Closed-form coordinate-wise update for a single weight coordinate.
        Uses Newton-Raphson style update for logistic regression.
        """
        n_samples, n_features = X.shape
        
        for class_idx in range(self.coef_.shape[0]):
            # Current predictions
            logits = X @ self.coef_[class_idx] + self.intercept_[class_idx]
            probs = self._sigmoid(logits)
            
            # Residuals
            if self.n_classes_ == 2:
                residuals = y_encoded.ravel() - probs
            else:
                residuals = y_encoded[:, class_idx] - probs
            
            # Compute gradient and Hessian diagonal for coordinate
            x_j = X[:, coord_idx]
            gradient = -np.dot(x_j, residuals) / n_samples + self.reg_lambda * self.coef_[class_idx, coord_idx]
            
            # Hessian diagonal (for logistic regression)
            hessian_diag = np.dot(x_j ** 2, probs * (1 - probs)) / n_samples + self.reg_lambda
            
            # Closed-form update
            if hessian_diag > 1e-10:
                self.coef_[class_idx, coord_idx] -= gradient / hessian_diag
    
    def _gradient_step(self, X, y_encoded):
        """Perform gradient descent step for all parameters."""
        n_samples = X.shape[0]
        logits = X @ self.coef_.T + self.intercept_
        
        if self.n_classes_ == 2:
            probs = self._sigmoid(logits.ravel())
            errors = probs - y_encoded.ravel()
            
            # Gradients
            grad_coef = np.outer(errors, X).reshape(1, -1) / n_samples + self.reg_lambda * self.coef_
            grad_intercept = np.array([np.mean(errors)])
        else:
            probs = self._softmax(logits)
            errors = probs - y_encoded
            
            # Gradients
            grad_coef = errors.T @ X / n_samples + self.reg_lambda * self.coef_
            grad_intercept = np.mean(errors, axis=0)
        
        # Update parameters
        self.coef_ -= self.learning_rate * grad_coef
        self.intercept_ -= self.learning_rate * grad_intercept
    
    def fit(self, X_train, y_train):
        """
        Fit the hybrid coordinate-gradient classifier.
        
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
        
        # Store classes
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        
        # Encode labels
        self.label_binarizer_ = LabelBinarizer()
        y_encoded = self.label_binarizer_.fit_transform(y_train)
        if self.n_classes_ == 2:
            y_encoded = y_encoded.ravel()
        
        # Initialize parameters
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X_train.shape
        
        if self.n_classes_ == 2:
            self.coef_ = rng.randn(1, n_features) * 0.01
            self.intercept_ = np.zeros(1)
        else:
            self.coef_ = rng.randn(self.n_classes_, n_features) * 0.01
            self.intercept_ = np.zeros(self.n_classes_)
        
        # Hybrid optimization loop
        prev_loss = float('inf')
        
        for iteration in range(self.n_iterations):
            # Phase 1: Coordinate-wise closed-form updates
            for _ in range(self.n_coord_updates):
                # Randomly select coordinates to update
                coord_indices = rng.choice(n_features, 
                                          size=min(n_features, max(1, n_features // 10)),
                                          replace=False)
                for coord_idx in coord_indices:
                    self._coordinate_update(X_train, y_encoded, coord_idx)
            
            # Phase 2: Gradient descent steps
            for _ in range(self.n_grad_steps):
                self._gradient_step(X_train, y_encoded)
            
            # Check convergence
            if iteration % 10 == 0:
                current_loss = self._compute_loss(X_train, y_encoded)
                
                if abs(prev_loss - current_loss) < self.tol:
                    break
                
                prev_loss = current_loss
        
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, 'is_fitted_')
        X_test = check_array(X_test)
        
        logits = X_test @ self.coef_.T + self.intercept_
        
        if self.n_classes_ == 2:
            probs = self._sigmoid(logits.ravel())
            return np.vstack([1 - probs, probs]).T
        else:
            return self._softmax(logits)
    
    def predict(self, X_test):
        """
        Predict class labels.
        
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
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]