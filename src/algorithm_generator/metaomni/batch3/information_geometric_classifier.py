import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.special import softmax, logsumexp


class InformationGeometricClassifier(BaseEstimator, ClassifierMixin):
    """
    Information Geometric Classifier using natural gradient descent.
    
    This classifier treats the parameter space as a Riemannian manifold where
    the metric is defined by the Fisher information matrix. It uses natural
    gradient descent to navigate this space, which accounts for the geometric
    structure of probability distributions.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for natural gradient descent.
    n_iterations : int, default=1000
        Maximum number of iterations for optimization.
    damping : float, default=1e-4
        Damping factor for Fisher information matrix regularization.
    tol : float, default=1e-4
        Tolerance for convergence.
    batch_size : int or None, default=None
        Batch size for stochastic natural gradient. If None, uses full batch.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, damping=1e-4,
                 tol=1e-4, batch_size=None, random_state=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.damping = damping
        self.tol = tol
        self.batch_size = batch_size
        self.random_state = random_state
    
    def _compute_fisher_information(self, X_bias, probs):
        """
        Compute the Fisher information matrix.
        
        The Fisher information defines the Riemannian metric on the
        statistical manifold of probability distributions.
        
        Parameters
        ----------
        X_bias : array-like of shape (n_samples, n_features + 1)
            Input features with bias term appended.
        probs : array-like of shape (n_samples, n_classes)
            Predicted probabilities for each class.
        
        Returns
        -------
        fisher : ndarray of shape (n_features + 1, n_features + 1)
            Fisher information matrix.
        """
        n_samples, n_features_bias = X_bias.shape
        n_classes = probs.shape[1]
        
        # Fisher information matrix approximation
        # For computational efficiency, we use the empirical Fisher
        fisher = np.zeros((n_features_bias, n_features_bias))
        
        for i in range(n_samples):
            x_i = X_bias[i]
            p_i = probs[i]
            
            # Compute outer product weighted by probability variance
            for c in range(n_classes):
                grad_c = p_i[c] * (1 - p_i[c])
                fisher += grad_c * np.outer(x_i, x_i)
        
        fisher /= n_samples
        
        # Add damping for numerical stability
        fisher += self.damping * np.eye(n_features_bias)
        
        return fisher
    
    def _natural_gradient(self, gradient, fisher):
        """
        Compute natural gradient by multiplying with inverse Fisher information.
        
        Natural gradient: ∇̃ = F^(-1) ∇
        where F is the Fisher information matrix.
        
        Parameters
        ----------
        gradient : ndarray
            Standard gradient vector.
        fisher : ndarray
            Fisher information matrix.
        
        Returns
        -------
        natural_grad : ndarray
            Natural gradient vector.
        """
        try:
            # Use Cholesky decomposition for stable inversion
            L = np.linalg.cholesky(fisher)
            natural_grad = np.linalg.solve(L.T, np.linalg.solve(L, gradient))
        except np.linalg.LinAlgError:
            # Fallback to regularized pseudo-inverse
            natural_grad = np.linalg.lstsq(fisher, gradient, rcond=None)[0]
        
        return natural_grad
    
    def _compute_description_length(self, X_bias, y, weights):
        """
        Compute the description length (negative log-likelihood + complexity).
        
        This serves as the objective function in the information geometric framework.
        
        Parameters
        ----------
        X_bias : array-like of shape (n_samples, n_features + 1)
            Input features with bias term.
        y : array-like of shape (n_samples,)
            Encoded target labels.
        weights : ndarray of shape (n_features + 1, n_classes)
            Model weights including bias.
        
        Returns
        -------
        description_length : float
            Total description length (loss + complexity).
        """
        n_samples = X_bias.shape[0]
        
        # Compute predictions
        logits = X_bias @ weights
        log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
        
        # Negative log-likelihood
        nll = -np.sum(log_probs[np.arange(n_samples), y]) / n_samples
        
        # Model complexity (L2 regularization as description length)
        complexity = 0.5 * self.damping * np.sum(weights ** 2)
        
        return nll + complexity
    
    def fit(self, X, y):
        """
        Fit the Information Geometric Classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input
        X, y = check_X_y(X, y)
        
        # Store classes
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Initialize weights (features + bias, classes)
        n_features = X.shape[1]
        self.weights_ = rng.randn(n_features + 1, self.n_classes_) * 0.01
        
        # Add bias column to X
        X_bias = np.column_stack([X, np.ones(X.shape[0])])
        
        # Natural gradient descent
        prev_loss = float('inf')
        
        for iteration in range(self.n_iterations):
            # Determine batch
            if self.batch_size is not None:
                indices = rng.choice(X.shape[0], size=min(self.batch_size, X.shape[0]), replace=False)
                X_batch = X_bias[indices]
                y_batch = y_encoded[indices]
            else:
                X_batch = X_bias
                y_batch = y_encoded
            
            # Forward pass
            logits = X_batch @ self.weights_
            probs = softmax(logits, axis=1)
            
            # Compute gradient of negative log-likelihood
            gradient = np.zeros_like(self.weights_)
            for c in range(self.n_classes_):
                y_c = (y_batch == c).astype(float)
                error = probs[:, c] - y_c
                gradient[:, c] = X_batch.T @ error / X_batch.shape[0]
            
            # Add regularization gradient
            gradient += self.damping * self.weights_
            
            # Compute Fisher information matrix
            fisher = self._compute_fisher_information(X_batch, probs)
            
            # Compute natural gradient for each class and update weights
            for c in range(self.n_classes_):
                nat_grad = self._natural_gradient(gradient[:, c], fisher)
                self.weights_[:, c] -= self.learning_rate * nat_grad
            
            # Check convergence every 10 iterations
            if iteration % 10 == 0:
                current_loss = self._compute_description_length(X_bias, y_encoded, self.weights_)
                
                if abs(prev_loss - current_loss) < self.tol:
                    break
                
                prev_loss = current_loss
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ['weights_', 'classes_'])
        X = check_array(X)
        
        # Add bias term
        X_bias = np.column_stack([X, np.ones(X.shape[0])])
        
        # Compute logits and probabilities
        logits = X_bias @ self.weights_
        probs = softmax(logits, axis=1)
        
        return probs
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        probs = self.predict_proba(X)
        y_pred_encoded = np.argmax(probs, axis=1)
        y_pred = self.label_encoder_.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True labels.
        
        Returns
        -------
        score : float
            Mean accuracy.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))