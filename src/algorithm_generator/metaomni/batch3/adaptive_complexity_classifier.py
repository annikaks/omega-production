import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import zlib
from typing import Literal


class AdaptiveComplexityClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive classifier that switches between discrete symbolic (decision tree) 
    and continuous embedding (neural network) representations based on local 
    Kolmogorov complexity estimates.
    
    Parameters
    ----------
    complexity_threshold : float, default=0.5
        Threshold for switching between discrete and continuous representations.
        Higher values favor discrete representations.
    
    window_size : int, default=10
        Number of nearest neighbors to consider for local complexity estimation.
    
    discrete_max_depth : int, default=10
        Maximum depth for the discrete decision tree classifier.
    
    continuous_hidden_layers : tuple, default=(100, 50)
        Hidden layer sizes for the continuous neural network classifier.
    
    strategy : {'adaptive', 'discrete', 'continuous'}, default='adaptive'
        Strategy for classification:
        - 'adaptive': dynamically switch based on complexity
        - 'discrete': always use discrete representation
        - 'continuous': always use continuous representation
    
    random_state : int, default=42
        Random state for reproducibility.
    """
    
    def __init__(
        self,
        complexity_threshold: float = 0.5,
        window_size: int = 10,
        discrete_max_depth: int = 10,
        continuous_hidden_layers: tuple = (100, 50),
        strategy: Literal['adaptive', 'discrete', 'continuous'] = 'adaptive',
        random_state: int = 42
    ):
        self.complexity_threshold = complexity_threshold
        self.window_size = window_size
        self.discrete_max_depth = discrete_max_depth
        self.continuous_hidden_layers = continuous_hidden_layers
        self.strategy = strategy
        self.random_state = random_state
    
    def _estimate_kolmogorov_complexity(self, data: np.ndarray) -> float:
        """
        Estimate Kolmogorov complexity using compression-based approximation.
        
        Parameters
        ----------
        data : np.ndarray
            Data to estimate complexity for.
        
        Returns
        -------
        float
            Normalized complexity estimate in [0, 1].
        """
        # Convert to bytes for compression
        data_bytes = data.tobytes()
        
        # Compress using zlib (approximation of Kolmogorov complexity)
        compressed = zlib.compress(data_bytes, level=9)
        
        # Normalize by original size
        complexity = len(compressed) / max(len(data_bytes), 1)
        
        return complexity
    
    def _compute_local_complexity(self, X: np.ndarray, indices: np.ndarray = None) -> np.ndarray:
        """
        Compute local Kolmogorov complexity for each sample.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
        indices : np.ndarray, optional
            Specific indices to compute complexity for.
        
        Returns
        -------
        np.ndarray
            Local complexity estimates for each sample.
        """
        if indices is None:
            indices = np.arange(X.shape[0])
        
        complexities = np.zeros(len(indices))
        
        for i, idx in enumerate(indices):
            # Find nearest neighbors
            distances = np.linalg.norm(X - X[idx], axis=1)
            nearest_indices = np.argsort(distances)[1:self.window_size + 1]
            
            # Get local neighborhood
            local_data = X[nearest_indices]
            
            # Estimate complexity of local region
            complexities[i] = self._estimate_kolmogorov_complexity(local_data.flatten())
        
        return complexities
    
    def _assign_representation(self, complexities: np.ndarray) -> np.ndarray:
        """
        Assign discrete (0) or continuous (1) representation based on complexity.
        
        Parameters
        ----------
        complexities : np.ndarray
            Local complexity estimates.
        
        Returns
        -------
        np.ndarray
            Binary array indicating representation type (0=discrete, 1=continuous).
        """
        # Low complexity -> discrete (symbolic)
        # High complexity -> continuous (embedding)
        return (complexities > self.complexity_threshold).astype(int)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the adaptive classifier.
        
        Parameters
        ----------
        X_train : np.ndarray of shape (n_samples, n_features)
            Training data.
        y_train : np.ndarray of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self
            Fitted classifier.
        """
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        self.classes_ = unique_labels(y_train)
        self.n_features_in_ = X_train.shape[1]
        
        # Store training data for complexity estimation
        self.X_train_ = X_train.copy()
        
        # Compute local complexity for training data
        if self.strategy == 'adaptive':
            self.train_complexities_ = self._compute_local_complexity(X_train)
            self.train_representations_ = self._assign_representation(self.train_complexities_)
            
            # Separate data by representation type
            discrete_mask = self.train_representations_ == 0
            continuous_mask = self.train_representations_ == 1
        elif self.strategy == 'discrete':
            discrete_mask = np.ones(X_train.shape[0], dtype=bool)
            continuous_mask = np.zeros(X_train.shape[0], dtype=bool)
        else:  # continuous
            discrete_mask = np.zeros(X_train.shape[0], dtype=bool)
            continuous_mask = np.ones(X_train.shape[0], dtype=bool)
        
        # Train discrete classifier (Decision Tree)
        self.discrete_classifier_ = DecisionTreeClassifier(
            max_depth=self.discrete_max_depth,
            random_state=self.random_state
        )
        
        if np.any(discrete_mask):
            self.discrete_classifier_.fit(X_train[discrete_mask], y_train[discrete_mask])
            self.has_discrete_ = True
        else:
            # Fit on all data as fallback
            self.discrete_classifier_.fit(X_train, y_train)
            self.has_discrete_ = False
        
        # Train continuous classifier (Neural Network)
        self.scaler_ = StandardScaler()
        X_train_scaled = self.scaler_.fit_transform(X_train)
        
        self.continuous_classifier_ = MLPClassifier(
            hidden_layer_sizes=self.continuous_hidden_layers,
            random_state=self.random_state,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        if np.any(continuous_mask):
            self.continuous_classifier_.fit(
                X_train_scaled[continuous_mask], 
                y_train[continuous_mask]
            )
            self.has_continuous_ = True
        else:
            # Fit on all data as fallback
            self.continuous_classifier_.fit(X_train_scaled, y_train)
            self.has_continuous_ = False
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : np.ndarray of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fitted
        check_is_fitted(self, ['discrete_classifier_', 'continuous_classifier_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_test.shape[1]}")
        
        # Compute local complexity for test data
        if self.strategy == 'adaptive':
            # Estimate complexity based on nearest neighbors in training data
            test_complexities = np.zeros(X_test.shape[0])
            
            for i in range(X_test.shape[0]):
                # Find nearest neighbors in training data
                distances = np.linalg.norm(self.X_train_ - X_test[i], axis=1)
                nearest_indices = np.argsort(distances)[:self.window_size]
                
                # Use average complexity of nearest training samples
                test_complexities[i] = np.mean(self.train_complexities_[nearest_indices])
            
            test_representations = self._assign_representation(test_complexities)
            
            # Separate predictions by representation type
            discrete_mask = test_representations == 0
            continuous_mask = test_representations == 1
        elif self.strategy == 'discrete':
            discrete_mask = np.ones(X_test.shape[0], dtype=bool)
            continuous_mask = np.zeros(X_test.shape[0], dtype=bool)
        else:  # continuous
            discrete_mask = np.zeros(X_test.shape[0], dtype=bool)
            continuous_mask = np.ones(X_test.shape[0], dtype=bool)
        
        # Initialize predictions
        predictions = np.zeros(X_test.shape[0], dtype=self.classes_.dtype)
        
        # Predict with discrete classifier
        if np.any(discrete_mask):
            predictions[discrete_mask] = self.discrete_classifier_.predict(X_test[discrete_mask])
        
        # Predict with continuous classifier
        if np.any(continuous_mask):
            X_test_scaled = self.scaler_.transform(X_test[continuous_mask])
            predictions[continuous_mask] = self.continuous_classifier_.predict(X_test_scaled)
        
        return predictions
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : np.ndarray of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self, ['discrete_classifier_', 'continuous_classifier_'])
        X_test = check_array(X_test)
        
        if self.strategy == 'adaptive':
            test_complexities = np.zeros(X_test.shape[0])
            
            for i in range(X_test.shape[0]):
                distances = np.linalg.norm(self.X_train_ - X_test[i], axis=1)
                nearest_indices = np.argsort(distances)[:self.window_size]
                test_complexities[i] = np.mean(self.train_complexities_[nearest_indices])
            
            test_representations = self._assign_representation(test_complexities)
            discrete_mask = test_representations == 0
            continuous_mask = test_representations == 1
        elif self.strategy == 'discrete':
            discrete_mask = np.ones(X_test.shape[0], dtype=bool)
            continuous_mask = np.zeros(X_test.shape[0], dtype=bool)
        else:
            discrete_mask = np.zeros(X_test.shape[0], dtype=bool)
            continuous_mask = np.ones(X_test.shape[0], dtype=bool)
        
        # Initialize probabilities
        probas = np.zeros((X_test.shape[0], len(self.classes_)))
        
        # Predict probabilities with discrete classifier
        if np.any(discrete_mask):
            probas[discrete_mask] = self.discrete_classifier_.predict_proba(X_test[discrete_mask])
        
        # Predict probabilities with continuous classifier
        if np.any(continuous_mask):
            X_test_scaled = self.scaler_.transform(X_test[continuous_mask])
            probas[continuous_mask] = self.continuous_classifier_.predict_proba(X_test_scaled)
        
        return probas