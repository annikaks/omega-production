import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelBinarizer
from collections import defaultdict


class StratifiedIndependentGradientClassifier(BaseEstimator, ClassifierMixin):
    """
    A neural network classifier that enforces statistical independence between
    gradient estimates using stratified sampling to maximize decorrelation
    across consecutive mini-batch updates.
    
    Parameters
    ----------
    hidden_dim : int, default=64
        Number of hidden units in the neural network.
    n_strata : int, default=10
        Number of strata to partition the dataset into for stratified sampling.
    batch_size : int, default=32
        Size of mini-batches for training.
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    n_epochs : int, default=100
        Number of training epochs.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, hidden_dim=64, n_strata=10, batch_size=32,
                 learning_rate=0.01, n_epochs=100, random_state=None):
        self.hidden_dim = hidden_dim
        self.n_strata = n_strata
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        
    def _sigmoid(self, x):
        """Sigmoid activation function with numerical stability."""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))
    
    def _softmax(self, x):
        """Softmax activation function with numerical stability."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _create_strata(self, X, y):
        """
        Create stratified partitions of the dataset to ensure decorrelation.
        Strata are created based on both class labels and feature space clustering.
        """
        n_samples = X.shape[0]
        strata = defaultdict(list)
        
        # Create strata based on class labels and feature quantiles
        for class_label in self.classes_:
            class_mask = (y == class_label)
            class_indices = np.where(class_mask)[0]
            
            if len(class_indices) == 0:
                continue
            
            # Further stratify within each class based on feature norms
            X_class = X[class_indices]
            feature_norms = np.linalg.norm(X_class, axis=1)
            
            # Divide into quantile-based strata
            n_strata_per_class = max(1, self.n_strata // len(self.classes_))
            quantiles = np.linspace(0, 100, n_strata_per_class + 1)
            
            for i in range(len(quantiles) - 1):
                lower = np.percentile(feature_norms, quantiles[i])
                upper = np.percentile(feature_norms, quantiles[i + 1])
                
                if i == len(quantiles) - 2:  # Last stratum includes upper bound
                    stratum_mask = (feature_norms >= lower) & (feature_norms <= upper)
                else:
                    stratum_mask = (feature_norms >= lower) & (feature_norms < upper)
                
                stratum_indices = class_indices[stratum_mask]
                if len(stratum_indices) > 0:
                    strata_key = f"{class_label}_{i}"
                    strata[strata_key] = stratum_indices.tolist()
        
        return strata
    
    def _sample_decorrelated_batch(self, strata, used_strata_history):
        """
        Sample a mini-batch that is maximally decorrelated from previous batches.
        Uses stratified sampling with stratum rotation to ensure independence.
        """
        batch_indices = []
        strata_keys = list(strata.keys())
        
        if len(strata_keys) == 0:
            return np.array([])
        
        # Score each stratum based on how recently it was used
        stratum_scores = {}
        for key in strata_keys:
            if len(strata[key]) == 0:
                continue
            # Higher score for strata used less recently
            last_used = used_strata_history.get(key, -np.inf)
            stratum_scores[key] = -last_used  # Negative so older = higher score
        
        if len(stratum_scores) == 0:
            return np.array([])
        
        # Sort strata by score (least recently used first)
        sorted_strata = sorted(stratum_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Sample from strata in order of priority
        samples_per_stratum = max(1, self.batch_size // len(sorted_strata))
        current_step = len(used_strata_history)
        
        for stratum_key, _ in sorted_strata:
            available_indices = strata[stratum_key]
            if len(available_indices) == 0:
                continue
                
            n_samples = min(samples_per_stratum, len(available_indices))
            sampled = self.rng_.choice(available_indices, size=n_samples, replace=False)
            batch_indices.extend(sampled)
            
            # Update usage history
            used_strata_history[stratum_key] = current_step
            
            if len(batch_indices) >= self.batch_size:
                break
        
        return np.array(batch_indices[:self.batch_size])
    
    def _forward(self, X):
        """Forward pass through the network."""
        self.hidden_ = self._sigmoid(np.dot(X, self.W1_) + self.b1_)
        output = np.dot(self.hidden_, self.W2_) + self.b2_
        self.output_ = self._softmax(output)
        return self.output_
    
    def _backward(self, X, y_encoded, batch_size):
        """Backward pass to compute gradients."""
        # Output layer gradients
        d_output = (self.output_ - y_encoded) / batch_size
        dW2 = np.dot(self.hidden_.T, d_output)
        db2 = np.sum(d_output, axis=0, keepdims=True)
        
        # Hidden layer gradients
        d_hidden = np.dot(d_output, self.W2_.T) * self.hidden_ * (1 - self.hidden_)
        dW1 = np.dot(X.T, d_hidden)
        db1 = np.sum(d_hidden, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier using stratified independent gradient descent.
        
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
        
        # Initialize random state
        self.rng_ = np.random.RandomState(self.random_state)
        
        # Initialize label binarizer
        self.label_binarizer_ = LabelBinarizer()
        y_encoded = self.label_binarizer_.fit_transform(y_train)
        if self.n_classes_ == 2:
            y_encoded = np.hstack([1 - y_encoded, y_encoded])
        
        # Initialize network parameters
        n_features = X_train.shape[1]
        scale1 = np.sqrt(2.0 / n_features)
        scale2 = np.sqrt(2.0 / self.hidden_dim)
        
        self.W1_ = self.rng_.randn(n_features, self.hidden_dim) * scale1
        self.b1_ = np.zeros((1, self.hidden_dim))
        self.W2_ = self.rng_.randn(self.hidden_dim, self.n_classes_) * scale2
        self.b2_ = np.zeros((1, self.n_classes_))
        
        # Create strata for stratified sampling
        strata = self._create_strata(X_train, y_train)
        
        # Training loop with decorrelated sampling
        used_strata_history = {}
        
        for epoch in range(self.n_epochs):
            # Reset strata usage periodically to allow revisiting
            if epoch % (self.n_epochs // 5 + 1) == 0:
                used_strata_history = {}
            
            # Refresh strata indices for each epoch
            epoch_strata = {k: v.copy() for k, v in strata.items()}
            
            n_batches = max(1, len(X_train) // self.batch_size)
            
            for _ in range(n_batches):
                # Sample decorrelated batch
                batch_indices = self._sample_decorrelated_batch(
                    epoch_strata, used_strata_history
                )
                
                if len(batch_indices) == 0:
                    break
                
                # Remove sampled indices from strata
                for key in epoch_strata:
                    epoch_strata[key] = [idx for idx in epoch_strata[key] 
                                        if idx not in batch_indices]
                
                X_batch = X_train[batch_indices]
                y_batch = y_encoded[batch_indices]
                
                # Forward and backward pass
                self._forward(X_batch)
                dW1, db1, dW2, db2 = self._backward(X_batch, y_batch, len(batch_indices))
                
                # Update parameters
                self.W1_ -= self.learning_rate * dW1
                self.b1_ -= self.learning_rate * db1
                self.W2_ -= self.learning_rate * dW2
                self.b2_ -= self.learning_rate * db2
        
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
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        return self._forward(X_test)
    
    def predict(self, X_test):
        """
        Predict class labels for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]