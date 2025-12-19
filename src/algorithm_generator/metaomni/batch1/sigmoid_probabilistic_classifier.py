import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler


class SigmoidProbabilisticClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses smooth sigmoid activation instead of hard thresholds
    and updates weights probabilistically based on prediction confidence.
    Supports both binary and multi-class classification using one-vs-rest strategy.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for weight updates.
    n_iterations : int, default=1000
        Number of training iterations.
    temperature : float, default=1.0
        Temperature parameter for sigmoid smoothness (lower = sharper).
    confidence_threshold : float, default=0.5
        Threshold for probabilistic weight updates based on confidence.
    random_state : int, default=None
        Random seed for reproducibility.
    learning_rate_decay : float, default=0.95
        Learning rate decay factor applied every 100 iterations.
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, 
                 temperature=1.0, confidence_threshold=0.5, 
                 random_state=None, learning_rate_decay=0.95):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.random_state = random_state
        self.learning_rate_decay = learning_rate_decay
    
    def _sigmoid(self, z):
        """Smooth sigmoid activation function with temperature scaling."""
        z_scaled = z / self.temperature
        return 1 / (1 + np.exp(-np.clip(z_scaled, -500, 500)))
    
    def _compute_confidence(self, probabilities):
        """
        Compute prediction confidence as distance from decision boundary (0.5).
        Returns values in [0, 1] where 1 is most confident.
        """
        return np.abs(probabilities - 0.5) * 2
    
    def _probabilistic_update_mask(self, confidence):
        """
        Generate probabilistic mask for weight updates based on confidence.
        Higher confidence = higher probability of update.
        """
        update_probs = confidence * (1 - self.confidence_threshold) + self.confidence_threshold
        return self.rng_.random(confidence.shape) < update_probs
    
    def _fit_binary(self, X_scaled, y_binary, n_features):
        """
        Fit a single binary classifier.
        
        Parameters
        ----------
        X_scaled : array-like of shape (n_samples, n_features)
            Scaled training data.
        y_binary : array-like of shape (n_samples,)
            Binary target values (0 or 1).
        n_features : int
            Number of features.
        
        Returns
        -------
        weights : ndarray of shape (n_features,)
            Fitted weights.
        bias : float
            Fitted bias term.
        """
        n_samples = X_scaled.shape[0]
        
        # Initialize weights and bias
        weights = self.rng_.randn(n_features) * 0.01
        bias = 0.0
        
        # Store initial learning rate
        current_lr = self.learning_rate
        
        # Training loop with probabilistic updates
        for iteration in range(self.n_iterations):
            # Forward pass
            linear_output = np.dot(X_scaled, weights) + bias
            predictions = self._sigmoid(linear_output)
            
            # Compute confidence for each prediction
            confidence = self._compute_confidence(predictions)
            
            # Generate probabilistic update mask
            update_mask = self._probabilistic_update_mask(confidence)
            
            # Compute gradients
            errors = predictions - y_binary
            
            # Apply probabilistic masking to gradients
            masked_errors = errors * update_mask
            
            # Update weights and bias
            dw = np.dot(X_scaled.T, masked_errors) / n_samples
            db = np.mean(masked_errors)
            
            weights -= current_lr * dw
            bias -= current_lr * db
            
            # Apply learning rate decay
            if iteration > 0 and iteration % 100 == 0:
                current_lr *= self.learning_rate_decay
        
        return weights, bias
    
    def fit(self, X, y):
        """
        Fit the classifier using probabilistic weight updates.
        Supports both binary and multi-class classification.
        
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
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        # Initialize random state
        self.rng_ = np.random.RandomState(self.random_state)
        
        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        n_samples, n_features = X_scaled.shape
        
        # Create label mapping
        self.label_map_ = {label: idx for idx, label in enumerate(self.classes_)}
        y_encoded = np.array([self.label_map_[label] for label in y])
        
        if self.n_classes_ == 2:
            # Binary classification
            y_binary = y_encoded
            self.weights_, self.bias_ = self._fit_binary(X_scaled, y_binary, n_features)
            self.is_binary_ = True
        else:
            # Multi-class classification using one-vs-rest
            self.weights_ = []
            self.bias_ = []
            self.is_binary_ = False
            
            for class_idx in range(self.n_classes_):
                # Create binary target: 1 if class_idx, 0 otherwise
                y_binary = (y_encoded == class_idx).astype(float)
                
                # Fit binary classifier for this class
                weights, bias = self._fit_binary(X_scaled, y_binary, n_features)
                self.weights_.append(weights)
                self.bias_.append(bias)
            
            # Convert to arrays for easier computation
            self.weights_ = np.array(self.weights_)  # shape: (n_classes, n_features)
            self.bias_ = np.array(self.bias_)  # shape: (n_classes,)
        
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
            Probability estimates for each class.
        """
        check_is_fitted(self, ['weights_', 'bias_', 'scaler_'])
        X = check_array(X)
        
        # Standardize features
        X_scaled = self.scaler_.transform(X)
        n_samples = X_scaled.shape[0]
        
        if self.is_binary_:
            # Binary classification
            linear_output = np.dot(X_scaled, self.weights_) + self.bias_
            prob_class_1 = self._sigmoid(linear_output)
            prob_class_0 = 1 - prob_class_1
            return np.column_stack([prob_class_0, prob_class_1])
        else:
            # Multi-class classification
            # Compute scores for all classes
            linear_outputs = np.dot(X_scaled, self.weights_.T) + self.bias_  # shape: (n_samples, n_classes)
            raw_probas = self._sigmoid(linear_outputs)
            
            # Normalize probabilities to sum to 1
            proba_sums = raw_probas.sum(axis=1, keepdims=True)
            proba_sums = np.where(proba_sums == 0, 1, proba_sums)  # Avoid division by zero
            probas = raw_probas / proba_sums
            
            return probas
    
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
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        
        # Map back to original labels
        return np.array([self.classes_[idx] for idx in class_indices])
    
    def get_confidence(self, X):
        """
        Get prediction confidence scores for X.
        For multi-class, returns the maximum probability as confidence.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        confidence : ndarray of shape (n_samples,)
            Confidence scores in [0, 1].
        """
        probabilities = self.predict_proba(X)
        
        if self.is_binary_:
            # For binary, use distance from decision boundary
            return self._compute_confidence(probabilities[:, 1])
        else:
            # For multi-class, use maximum probability as confidence
            return np.max(probabilities, axis=1)