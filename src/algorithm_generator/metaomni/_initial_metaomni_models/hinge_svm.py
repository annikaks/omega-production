import numpy as np
from scipy.optimize import minimize

class HingeSVM:
    def __init__(self, C=1.0, max_iter=1000, tol=1e-3):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None

    def _hinge_loss(self, w, X, y):
        margin = y * (np.dot(X, w[:-1]) + w[-1])
        loss = np.maximum(0, 1 - margin)
        return np.sum(loss) / len(y) + 0.5 * self.C * np.dot(w[:-1], w[:-1])

    def _hinge_loss_gradient(self, w, X, y):
        margin = y * (np.dot(X, w[:-1]) + w[-1])
        mask = (margin < 1).astype(int)
        grad_w = -np.mean(mask[:, np.newaxis] * y[:, np.newaxis] * X, axis=0) + self.C * w[:-1]
        grad_b = -np.mean(mask * y)
        return np.concatenate([grad_w, [grad_b]])

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        
        # Initialize weights and bias
        initial_w = np.zeros(n_features + 1)
        
        # Define the optimization problem
        def objective(w):
            return self._hinge_loss(w, X_train, y_train)
        
        def gradient(w):
            return self._hinge_loss_gradient(w, X_train, y_train)
        
        # Optimize using L-BFGS-B algorithm
        result = minimize(
            objective,
            initial_w,
            method='L-BFGS-B',
            jac=gradient,
            options={'maxiter': self.max_iter, 'gtol': self.tol}
        )
        
        # Extract optimized weights and bias
        self.w = result.x[:-1]
        self.b = result.x[-1]
        
        return self

    def predict(self, X_test):
        if self.w is None or self.b is None:
            raise ValueError("Model has not been fitted. Call 'fit' before making predictions.")
        
        scores = np.dot(X_test, self.w) + self.b
        return np.sign(scores)