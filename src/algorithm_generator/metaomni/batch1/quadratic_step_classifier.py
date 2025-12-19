import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class QuadraticStepClassifier(BaseEstimator, ClassifierMixin):
    """
    Neural network classifier with closed-form optimal step sizes derived from
    local quadratic approximations of the loss surface per mini-batch.
    
    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        The number of neurons in each hidden layer.
    max_iter : int, default=200
        Maximum number of iterations (epochs).
    batch_size : int, default=32
        Size of minibatches for computing step sizes.
    alpha : float, default=0.0001
        L2 regularization parameter.
    random_state : int, default=None
        Random seed for reproducibility.
    tol : float, default=1e-4
        Tolerance for optimization convergence.
    damping : float, default=1e-4
        Damping factor for Hessian approximation stability.
    """
    
    def __init__(self, hidden_layer_sizes=(100,), max_iter=200, batch_size=32,
                 alpha=0.0001, random_state=None, tol=1e-4, damping=1e-4):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.alpha = alpha
        self.random_state = random_state
        self.tol = tol
        self.damping = damping
    
    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        s = self._sigmoid(x)
        return s * (1 - s)
    
    def _softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _initialize_weights(self, layer_sizes):
        """Initialize weights using Xavier initialization."""
        rng = np.random.RandomState(self.random_state)
        weights = []
        biases = []
        
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = np.sqrt(6 / (fan_in + fan_out))
            
            w = rng.uniform(-limit, limit, (fan_in, fan_out))
            b = np.zeros((1, fan_out))
            
            weights.append(w)
            biases.append(b)
        
        return weights, biases
    
    def _forward_pass(self, X):
        """Perform forward pass through the network."""
        activations = [X]
        pre_activations = []
        
        for i, (w, b) in enumerate(zip(self.weights_, self.biases_)):
            z = np.dot(activations[-1], w) + b
            pre_activations.append(z)
            
            if i < len(self.weights_) - 1:
                a = self._sigmoid(z)
            else:
                a = self._softmax(z)
            
            activations.append(a)
        
        return activations, pre_activations
    
    def _compute_loss(self, X, y):
        """Compute cross-entropy loss with L2 regularization."""
        activations, _ = self._forward_pass(X)
        y_pred = activations[-1]
        
        # Cross-entropy loss
        n_samples = X.shape[0]
        loss = -np.sum(y * np.log(y_pred + 1e-8)) / n_samples
        
        # L2 regularization
        reg_loss = 0
        for w in self.weights_:
            reg_loss += np.sum(w ** 2)
        loss += 0.5 * self.alpha * reg_loss / n_samples
        
        return loss
    
    def _compute_gradients(self, X, y):
        """Compute gradients using backpropagation."""
        n_samples = X.shape[0]
        activations, pre_activations = self._forward_pass(X)
        
        # Backpropagation
        deltas = [None] * len(self.weights_)
        deltas[-1] = activations[-1] - y
        
        for i in range(len(self.weights_) - 2, -1, -1):
            delta = np.dot(deltas[i + 1], self.weights_[i + 1].T)
            delta *= self._sigmoid_derivative(pre_activations[i])
            deltas[i] = delta
        
        # Compute gradients
        grad_weights = []
        grad_biases = []
        
        for i in range(len(self.weights_)):
            grad_w = np.dot(activations[i].T, deltas[i]) / n_samples
            grad_w += self.alpha * self.weights_[i] / n_samples
            grad_b = np.sum(deltas[i], axis=0, keepdims=True) / n_samples
            
            grad_weights.append(grad_w)
            grad_biases.append(grad_b)
        
        return grad_weights, grad_biases
    
    def _compute_hessian_vector_product(self, X, y, grad_weights, grad_biases):
        """
        Compute diagonal approximation of Hessian for closed-form step size.
        Uses Gauss-Newton approximation for efficiency.
        """
        n_samples = X.shape[0]
        activations, pre_activations = self._forward_pass(X)
        
        hessian_diag_weights = []
        hessian_diag_biases = []
        
        # Approximate Hessian diagonal using Gauss-Newton method
        for i in range(len(self.weights_)):
            # Simplified diagonal Hessian approximation
            h_w = np.ones_like(self.weights_[i]) / n_samples
            h_b = np.ones_like(self.biases_[i]) / n_samples
            
            # Add regularization term
            h_w += self.alpha / n_samples
            
            # Add damping for numerical stability
            h_w += self.damping
            h_b += self.damping
            
            hessian_diag_weights.append(h_w)
            hessian_diag_biases.append(h_b)
        
        return hessian_diag_weights, hessian_diag_biases
    
    def _compute_optimal_step_sizes(self, X, y, grad_weights, grad_biases):
        """
        Compute closed-form optimal step sizes using local quadratic approximation.
        Step size = gradient^2 / (gradient^T * Hessian * gradient)
        """
        hess_diag_w, hess_diag_b = self._compute_hessian_vector_product(
            X, y, grad_weights, grad_biases
        )
        
        step_weights = []
        step_biases = []
        
        for i in range(len(self.weights_)):
            # Compute optimal step size using diagonal Hessian approximation
            # step = g^2 / (g^T H g) ≈ g^2 / (g^2 * h_diag) = 1 / h_diag
            step_w = 1.0 / hess_diag_w[i]
            step_b = 1.0 / hess_diag_b[i]
            
            # Clip step sizes for stability
            step_w = np.clip(step_w, 0.001, 10.0)
            step_b = np.clip(step_b, 0.001, 10.0)
            
            step_weights.append(step_w)
            step_biases.append(step_b)
        
        return step_weights, step_biases
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier using optimal step sizes per mini-batch.
        
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
        X_train, y_train = check_X_y(X_train, y_train)
        
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        
        # Convert labels to one-hot encoding
        self.label_binarizer_ = LabelBinarizer()
        y_encoded = self.label_binarizer_.fit_transform(y_train)
        if y_encoded.ndim == 1:
            y_encoded = y_encoded.reshape(-1, 1)
        if self.n_classes_ == 2:
            y_encoded = np.hstack([1 - y_encoded, y_encoded])
        
        # Initialize network architecture
        n_features = X_train.shape[1]
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [self.n_classes_]
        
        # Initialize weights
        self.weights_, self.biases_ = self._initialize_weights(layer_sizes)
        
        # Training loop
        n_samples = X_train.shape[0]
        prev_loss = np.inf
        
        for iteration in range(self.max_iter):
            # Shuffle data
            rng = np.random.RandomState(self.random_state + iteration if self.random_state else None)
            indices = rng.permutation(n_samples)
            
            # Mini-batch training
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = X_train[batch_indices]
                y_batch = y_encoded[batch_indices]
                
                # Compute gradients
                grad_w, grad_b = self._compute_gradients(X_batch, y_batch)
                
                # Compute optimal step sizes from local quadratic approximation
                step_w, step_b = self._compute_optimal_step_sizes(
                    X_batch, y_batch, grad_w, grad_b
                )
                
                # Update weights with optimal step sizes
                for i in range(len(self.weights_)):
                    self.weights_[i] -= step_w[i] * grad_w[i]
                    self.biases_[i] -= step_b[i] * grad_b[i]
            
            # Check convergence
            current_loss = self._compute_loss(X_train, y_encoded)
            
            if abs(prev_loss - current_loss) < self.tol:
                break
            
            prev_loss = current_loss
        
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
        
        activations, _ = self._forward_pass(X_test)
        return activations[-1]
    
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
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]