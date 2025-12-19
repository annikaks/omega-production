import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelBinarizer


class StochasticEnsembleGradientClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that injects controlled stochastic perturbations into gradient
    calculations and combines multiple perturbed gradient paths through ensemble
    averaging for robust convergence.
    
    Parameters
    ----------
    n_estimators : int, default=5
        Number of perturbed gradient paths in the ensemble.
    
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    
    n_iterations : int, default=1000
        Number of gradient descent iterations.
    
    perturbation_scale : float, default=0.1
        Scale of stochastic perturbations injected into gradients.
    
    perturbation_type : str, default='gaussian'
        Type of perturbation: 'gaussian', 'uniform', or 'laplace'.
    
    regularization : float, default=0.01
        L2 regularization parameter.
    
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, n_estimators=5, learning_rate=0.01, n_iterations=1000,
                 perturbation_scale=0.1, perturbation_type='gaussian',
                 regularization=0.01, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.perturbation_scale = perturbation_scale
        self.perturbation_type = perturbation_type
        self.regularization = regularization
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
    
    def _generate_perturbation(self, shape, rng):
        """Generate stochastic perturbation based on perturbation type."""
        if self.perturbation_type == 'gaussian':
            return rng.normal(0, self.perturbation_scale, shape)
        elif self.perturbation_type == 'uniform':
            return rng.uniform(-self.perturbation_scale, self.perturbation_scale, shape)
        elif self.perturbation_type == 'laplace':
            return rng.laplace(0, self.perturbation_scale, shape)
        else:
            raise ValueError(f"Unknown perturbation type: {self.perturbation_type}")
    
    def _compute_gradient(self, X, y, weights, bias, rng):
        """Compute gradient with stochastic perturbations."""
        n_samples = X.shape[0]
        
        # Forward pass
        logits = np.dot(X, weights) + bias
        
        if self.n_classes_ == 2:
            predictions = self._sigmoid(logits).ravel()
            error = predictions - y
            
            # Compute base gradient
            grad_weights = np.dot(X.T, error) / n_samples
            grad_bias = np.mean(error)
        else:
            predictions = self._softmax(logits)
            error = predictions - y
            
            # Compute base gradient
            grad_weights = np.dot(X.T, error) / n_samples
            grad_bias = np.mean(error, axis=0)
        
        # Add L2 regularization
        grad_weights += self.regularization * weights
        
        # Inject stochastic perturbations
        perturb_weights = self._generate_perturbation(grad_weights.shape, rng)
        perturb_bias = self._generate_perturbation(grad_bias.shape, rng)
        
        grad_weights += perturb_weights
        grad_bias += perturb_bias
        
        return grad_weights, grad_bias
    
    def _train_single_path(self, X, y, rng):
        """Train a single perturbed gradient path."""
        n_features = X.shape[1]
        
        # Initialize weights
        if self.n_classes_ == 2:
            weights = rng.normal(0, 0.01, (n_features, 1))
            bias = np.zeros(1)
        else:
            weights = rng.normal(0, 0.01, (n_features, self.n_classes_))
            bias = np.zeros(self.n_classes_)
        
        # Gradient descent with perturbations
        for iteration in range(self.n_iterations):
            grad_weights, grad_bias = self._compute_gradient(X, y, weights, bias, rng)
            
            # Update parameters
            weights -= self.learning_rate * grad_weights
            bias -= self.learning_rate * grad_bias
        
        return weights, bias
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier using ensemble of perturbed gradient paths.
        
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
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        
        # Encode labels
        self.label_binarizer_ = LabelBinarizer()
        if self.n_classes_ == 2:
            y_encoded = self.label_binarizer_.fit_transform(y_train).ravel()
        else:
            y_encoded = self.label_binarizer_.fit_transform(y_train)
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Train ensemble of perturbed gradient paths
        self.ensemble_weights_ = []
        self.ensemble_biases_ = []
        
        for i in range(self.n_estimators):
            # Create separate RNG for each estimator
            estimator_rng = np.random.RandomState(
                None if self.random_state is None else self.random_state + i
            )
            
            weights, bias = self._train_single_path(X_train, y_encoded, estimator_rng)
            self.ensemble_weights_.append(weights)
            self.ensemble_biases_.append(bias)
        
        # Compute ensemble-averaged weights and biases
        self.weights_ = np.mean(self.ensemble_weights_, axis=0)
        self.bias_ = np.mean(self.ensemble_biases_, axis=0)
        
        self.is_fitted_ = True
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
        check_is_fitted(self, 'is_fitted_')
        X_test = check_array(X_test)
        
        # Compute predictions using ensemble-averaged parameters
        logits = np.dot(X_test, self.weights_) + self.bias_
        
        if self.n_classes_ == 2:
            proba_positive = self._sigmoid(logits).ravel()
            return np.vstack([1 - proba_positive, proba_positive]).T
        else:
            return self._softmax(logits)
    
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
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]