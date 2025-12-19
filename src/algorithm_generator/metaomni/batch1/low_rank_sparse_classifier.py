import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelBinarizer


class LowRankSparseClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that decomposes weight vectors into low-rank plus sparse components.
    
    The weight matrix W is decomposed as: W = L + S
    where L is low-rank (capturing systematic patterns across features)
    and S is sparse (capturing feature-specific adjustments).
    
    Parameters
    ----------
    rank : int, default=5
        Rank of the low-rank component L.
    alpha : float, default=1.0
        Regularization strength for the sparse component (L1 penalty).
    beta : float, default=1.0
        Regularization strength for the low-rank component (Frobenius norm).
    max_iter : int, default=1000
        Maximum number of iterations for optimization.
    tol : float, default=1e-4
        Tolerance for convergence.
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, rank=5, alpha=1.0, beta=1.0, max_iter=1000, 
                 tol=1e-4, learning_rate=0.01, random_state=None):
        self.rank = rank
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.random_state = random_state
    
    def _sigmoid(self, z):
        """Numerically stable sigmoid function."""
        return np.where(z >= 0, 
                       1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))
    
    def _soft_threshold(self, x, threshold):
        """Soft thresholding operator for L1 regularization."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def _compute_loss(self, X, y, U, V, S, b):
        """Compute the total loss including regularization terms."""
        L = U @ V.T
        W = L + S
        logits = X @ W + b
        
        # Binary cross-entropy loss
        predictions = self._sigmoid(logits)
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        
        bce_loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        
        # Regularization terms
        l1_penalty = self.alpha * np.sum(np.abs(S))
        frobenius_penalty = self.beta * (np.sum(U**2) + np.sum(V**2))
        
        return bce_loss + l1_penalty + frobenius_penalty
    
    def fit(self, X, y):
        """
        Fit the low-rank plus sparse classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)
        
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        # Convert labels to binary format
        self.label_binarizer_ = LabelBinarizer()
        y_bin = self.label_binarizer_.fit_transform(y)
        
        if self.n_classes_ == 2:
            y_bin = y_bin.ravel()
            n_outputs = 1
        else:
            n_outputs = self.n_classes_
        
        n_samples, n_features = X.shape
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Initialize parameters
        # Low-rank component: L = U @ V.T where U is (n_features, rank) and V is (n_outputs, rank)
        scale = 0.01
        self.U_ = rng.randn(n_features, self.rank) * scale
        self.V_ = rng.randn(n_outputs, self.rank) * scale
        
        # Sparse component
        self.S_ = np.zeros((n_features, n_outputs))
        
        # Bias term
        self.b_ = np.zeros(n_outputs)
        
        # Optimization loop using alternating minimization
        prev_loss = float('inf')
        
        for iteration in range(self.max_iter):
            # Compute current low-rank component
            L = self.U_ @ self.V_.T
            W = L + self.S_
            
            # Forward pass
            logits = X @ W + self.b_
            predictions = self._sigmoid(logits)
            
            # Compute gradients
            error = predictions - y_bin.reshape(-1, n_outputs)
            
            # Gradient w.r.t. W
            grad_W = (X.T @ error) / n_samples
            
            # Gradient w.r.t. bias
            grad_b = np.mean(error, axis=0)
            
            # Update U (low-rank component)
            grad_U = grad_W @ self.V_ + 2 * self.beta * self.U_
            self.U_ -= self.learning_rate * grad_U
            
            # Update V (low-rank component)
            grad_V = grad_W.T @ self.U_ + 2 * self.beta * self.V_
            self.V_ -= self.learning_rate * grad_V
            
            # Update S (sparse component) with soft thresholding
            L = self.U_ @ self.V_.T
            S_update = self.S_ - self.learning_rate * grad_W
            self.S_ = self._soft_threshold(S_update, self.learning_rate * self.alpha)
            
            # Update bias
            self.b_ -= self.learning_rate * grad_b
            
            # Check convergence
            if iteration % 10 == 0:
                current_loss = self._compute_loss(X, y_bin.reshape(-1, n_outputs), 
                                                  self.U_, self.V_, self.S_, self.b_)
                
                if abs(prev_loss - current_loss) < self.tol:
                    break
                
                prev_loss = current_loss
        
        # Store final weight matrix
        self.L_ = self.U_ @ self.V_.T
        self.W_ = self.L_ + self.S_
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ['W_', 'b_'])
        X = check_array(X)
        
        logits = X @ self.W_ + self.b_
        
        if self.n_classes_ == 2:
            proba_positive = self._sigmoid(logits).ravel()
            return np.vstack([1 - proba_positive, proba_positive]).T
        else:
            # Softmax for multiclass
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def predict(self, X):
        """
        Predict class labels for X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]
    
    def get_components(self):
        """
        Get the low-rank and sparse components.
        
        Returns
        -------
        components : dict
            Dictionary containing 'low_rank' (L), 'sparse' (S), and 'total' (W) weight matrices.
        """
        check_is_fitted(self, ['L_', 'S_', 'W_'])
        return {
            'low_rank': self.L_,
            'sparse': self.S_,
            'total': self.W_,
            'U': self.U_,
            'V': self.V_
        }