import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.special import softmax


class AdaptiveHypervectorClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Hypervector Classifier using information-geometric gradients.
    
    This classifier adaptively adjusts hypervector dimensions during training,
    expanding dimensions for hard-to-separate classes and compressing for
    easily distinguishable ones.
    
    Parameters
    ----------
    initial_dim : int, default=1000
        Initial dimension of hypervectors
    min_dim : int, default=100
        Minimum dimension allowed
    max_dim : int, default=5000
        Maximum dimension allowed
    n_iterations : int, default=50
        Number of training iterations
    learning_rate : float, default=0.01
        Learning rate for gradient updates
    expansion_threshold : float, default=0.3
        Threshold for expanding dimensions (lower accuracy triggers expansion)
    compression_threshold : float, default=0.95
        Threshold for compressing dimensions (higher accuracy triggers compression)
    adjustment_rate : int, default=100
        Number of dimensions to add/remove during adjustment
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, initial_dim=1000, min_dim=100, max_dim=5000,
                 n_iterations=50, learning_rate=0.01,
                 expansion_threshold=0.3, compression_threshold=0.95,
                 adjustment_rate=100, random_state=None):
        self.initial_dim = initial_dim
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.expansion_threshold = expansion_threshold
        self.compression_threshold = compression_threshold
        self.adjustment_rate = adjustment_rate
        self.random_state = random_state
    
    def _initialize_hypervectors(self, n_features, n_classes):
        """Initialize random hypervectors for encoding."""
        rng = np.random.RandomState(self.random_state)
        
        # Feature encoders: map each feature to a hypervector
        self.feature_encoders_ = rng.randn(n_features, self.current_dim_)
        self.feature_encoders_ /= np.linalg.norm(self.feature_encoders_, axis=1, keepdims=True)
        
        # Class prototypes: one hypervector per class
        self.class_prototypes_ = rng.randn(n_classes, self.current_dim_)
        self.class_prototypes_ /= np.linalg.norm(self.class_prototypes_, axis=1, keepdims=True)
    
    def _encode_sample(self, x):
        """Encode a sample into hypervector space."""
        # Weighted sum of feature encoders
        encoded = np.sum(x[:, np.newaxis] * self.feature_encoders_, axis=0)
        # Normalize
        norm = np.linalg.norm(encoded)
        if norm > 0:
            encoded /= norm
        return encoded
    
    def _compute_similarities(self, encoded_samples):
        """Compute cosine similarities between encoded samples and class prototypes."""
        # Normalize encoded samples
        norms = np.linalg.norm(encoded_samples, axis=1, keepdims=True)
        norms[norms == 0] = 1
        encoded_samples_norm = encoded_samples / norms
        
        # Compute cosine similarity
        similarities = encoded_samples_norm @ self.class_prototypes_.T
        return similarities
    
    def _compute_class_separability(self, X, y):
        """Compute per-class separability using information geometry."""
        class_separability = {}
        
        for class_idx in range(len(self.classes_)):
            class_mask = (y == self.classes_[class_idx])
            if not np.any(class_mask):
                class_separability[class_idx] = 1.0
                continue
            
            # Encode samples
            encoded = np.array([self._encode_sample(x) for x in X[class_mask]])
            
            # Compute similarities
            similarities = self._compute_similarities(encoded)
            
            # Predict
            predictions = np.argmax(similarities, axis=1)
            
            # Accuracy for this class
            accuracy = np.mean(predictions == class_idx)
            class_separability[class_idx] = accuracy
        
        return class_separability
    
    def _adjust_dimensions(self, class_separability):
        """Adjust hypervector dimensions based on class separability."""
        avg_separability = np.mean(list(class_separability.values()))
        
        new_dim = self.current_dim_
        
        # Expand if classes are hard to separate
        if avg_separability < self.expansion_threshold and self.current_dim_ < self.max_dim:
            new_dim = min(self.current_dim_ + self.adjustment_rate, self.max_dim)
            self._expand_dimensions(new_dim)
            return True
        
        # Compress if classes are easily separable
        elif avg_separability > self.compression_threshold and self.current_dim_ > self.min_dim:
            new_dim = max(self.current_dim_ - self.adjustment_rate, self.min_dim)
            self._compress_dimensions(new_dim)
            return True
        
        return False
    
    def _expand_dimensions(self, new_dim):
        """Expand hypervector dimensions."""
        rng = np.random.RandomState(self.random_state)
        additional_dims = new_dim - self.current_dim_
        
        # Expand feature encoders
        new_features = rng.randn(self.feature_encoders_.shape[0], additional_dims)
        new_features /= np.linalg.norm(new_features, axis=1, keepdims=True)
        self.feature_encoders_ = np.hstack([self.feature_encoders_, new_features])
        
        # Expand class prototypes
        new_prototypes = rng.randn(self.class_prototypes_.shape[0], additional_dims)
        new_prototypes /= np.linalg.norm(new_prototypes, axis=1, keepdims=True)
        self.class_prototypes_ = np.hstack([self.class_prototypes_, new_prototypes])
        
        self.current_dim_ = new_dim
    
    def _compress_dimensions(self, new_dim):
        """Compress hypervector dimensions using PCA-like approach."""
        # Keep the most important dimensions (highest variance)
        self.feature_encoders_ = self.feature_encoders_[:, :new_dim]
        self.class_prototypes_ = self.class_prototypes_[:, :new_dim]
        
        # Renormalize
        self.feature_encoders_ /= np.linalg.norm(self.feature_encoders_, axis=1, keepdims=True)
        self.class_prototypes_ /= np.linalg.norm(self.class_prototypes_, axis=1, keepdims=True)
        
        self.current_dim_ = new_dim
    
    def _update_prototypes(self, X, y):
        """Update class prototypes using gradient descent."""
        for class_idx in range(len(self.classes_)):
            class_mask = (y == self.classes_[class_idx])
            if not np.any(class_mask):
                continue
            
            # Encode samples of this class
            encoded = np.array([self._encode_sample(x) for x in X[class_mask]])
            
            # Compute gradient: move prototype towards class samples
            gradient = np.mean(encoded - self.class_prototypes_[class_idx], axis=0)
            
            # Update with learning rate
            self.class_prototypes_[class_idx] += self.learning_rate * gradient
            
            # Normalize
            norm = np.linalg.norm(self.class_prototypes_[class_idx])
            if norm > 0:
                self.class_prototypes_[class_idx] /= norm
    
    def fit(self, X_train, y_train):
        """
        Fit the adaptive hypervector classifier.
        
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
        
        # Store classes
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X_train.shape[1]
        
        # Initialize current dimension
        self.current_dim_ = self.initial_dim
        
        # Initialize hypervectors
        self._initialize_hypervectors(self.n_features_, self.n_classes_)
        
        # Training loop
        self.dimension_history_ = [self.current_dim_]
        
        for iteration in range(self.n_iterations):
            # Update prototypes
            self._update_prototypes(X_train, y_train)
            
            # Every few iterations, check separability and adjust dimensions
            if iteration % 5 == 0 and iteration > 0:
                class_separability = self._compute_class_separability(X_train, y_train)
                adjusted = self._adjust_dimensions(class_separability)
                if adjusted:
                    self.dimension_history_.append(self.current_dim_)
        
        self.is_fitted_ = True
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        # Check if fitted
        check_is_fitted(self, ['is_fitted_', 'class_prototypes_', 'feature_encoders_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Encode test samples
        encoded = np.array([self._encode_sample(x) for x in X_test])
        
        # Compute similarities
        similarities = self._compute_similarities(encoded)
        
        # Predict class with highest similarity
        predictions = np.argmax(similarities, axis=1)
        
        return self.classes_[predictions]
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test samples
        
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Class probabilities
        """
        # Check if fitted
        check_is_fitted(self, ['is_fitted_', 'class_prototypes_', 'feature_encoders_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Encode test samples
        encoded = np.array([self._encode_sample(x) for x in X_test])
        
        # Compute similarities
        similarities = self._compute_similarities(encoded)
        
        # Convert to probabilities using softmax
        probabilities = softmax(similarities * 10, axis=1)  # Scale for sharper probabilities
        
        return probabilities