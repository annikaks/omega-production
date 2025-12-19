import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder


class HybridHDClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid Hyperdimensional Computing Classifier.
    
    Uses binding operations for categorical features and smooth rotations
    for continuous features in hyperdimensional space.
    
    Parameters
    ----------
    n_dimensions : int, default=10000
        Dimensionality of hypervectors
    categorical_indices : list or None, default=None
        Indices of categorical features. If None, all features treated as continuous.
    n_levels : int, default=100
        Number of quantization levels for continuous features
    random_state : int or None, default=42
        Random seed for reproducibility
    """
    
    def __init__(self, n_dimensions=10000, categorical_indices=None, 
                 n_levels=100, random_state=42):
        self.n_dimensions = n_dimensions
        self.categorical_indices = categorical_indices
        self.n_levels = n_levels
        self.random_state = random_state
        
    def _generate_random_hv(self, shape=None):
        """Generate random bipolar hypervector(s)."""
        rng = np.random.RandomState(self.random_state + getattr(self, '_seed_offset', 0))
        if shape is None:
            shape = self.n_dimensions
        return rng.choice([-1, 1], size=shape)
    
    def _bind(self, hv1, hv2):
        """Bind two hypervectors using element-wise multiplication."""
        return hv1 * hv2
    
    def _bundle(self, hvs):
        """Bundle multiple hypervectors using element-wise addition and thresholding."""
        summed = np.sum(hvs, axis=0)
        return np.sign(summed + (summed == 0) * self._generate_random_hv())
    
    def _rotate(self, hv, angle):
        """
        Simulate rotation in HD space using permutation.
        Angle in [0, 1] determines the amount of rotation.
        """
        n_shifts = int(angle * self.n_dimensions)
        return np.roll(hv, n_shifts)
    
    def _create_level_hvs(self, n_levels):
        """Create hypervectors for quantization levels using smooth rotations."""
        base_hv = self._generate_random_hv()
        level_hvs = []
        for i in range(n_levels):
            angle = i / n_levels
            level_hvs.append(self._rotate(base_hv, angle))
        return np.array(level_hvs)
    
    def _encode_categorical(self, value, feature_idx):
        """Encode categorical feature using binding."""
        # Get or create feature base hypervector
        if feature_idx not in self.feature_hvs_:
            self._seed_offset = feature_idx * 1000
            self.feature_hvs_[feature_idx] = self._generate_random_hv()
        
        # Get or create value hypervector
        key = (feature_idx, value)
        if key not in self.value_hvs_:
            self._seed_offset = hash(key) % 1000000
            self.value_hvs_[key] = self._generate_random_hv()
        
        # Bind feature and value hypervectors
        return self._bind(self.feature_hvs_[feature_idx], self.value_hvs_[key])
    
    def _encode_continuous(self, value, feature_idx, min_val, max_val):
        """Encode continuous feature using smooth rotation."""
        # Normalize value to [0, 1]
        if max_val > min_val:
            normalized = (value - min_val) / (max_val - min_val)
        else:
            normalized = 0.5
        normalized = np.clip(normalized, 0, 1)
        
        # Quantize to nearest level
        level = int(normalized * (self.n_levels - 1))
        level = np.clip(level, 0, self.n_levels - 1)
        
        # Get or create feature base hypervector
        if feature_idx not in self.feature_hvs_:
            self._seed_offset = feature_idx * 1000
            self.feature_hvs_[feature_idx] = self._generate_random_hv()
        
        # Bind with rotated level hypervector
        level_hv = self.level_hvs_[level]
        return self._bind(self.feature_hvs_[feature_idx], level_hv)
    
    def _encode_sample(self, x):
        """Encode a single sample into hyperdimensional space."""
        feature_hvs = []
        
        for idx, value in enumerate(x):
            if self.categorical_mask_[idx]:
                # Categorical feature
                hv = self._encode_categorical(value, idx)
            else:
                # Continuous feature
                min_val = self.feature_ranges_[idx][0]
                max_val = self.feature_ranges_[idx][1]
                hv = self._encode_continuous(value, idx, min_val, max_val)
            feature_hvs.append(hv)
        
        # Bundle all feature hypervectors
        return self._bundle(feature_hvs)
    
    def fit(self, X, y):
        """
        Fit the hybrid HD classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        X, y = check_X_y(X, y)
        
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        # Initialize storage
        self.feature_hvs_ = {}
        self.value_hvs_ = {}
        self.class_hvs_ = {}
        
        # Determine which features are categorical
        if self.categorical_indices is None:
            self.categorical_mask_ = np.zeros(self.n_features_in_, dtype=bool)
        else:
            self.categorical_mask_ = np.zeros(self.n_features_in_, dtype=bool)
            self.categorical_mask_[self.categorical_indices] = True
        
        # Compute feature ranges for continuous features
        self.feature_ranges_ = {}
        for idx in range(self.n_features_in_):
            if not self.categorical_mask_[idx]:
                self.feature_ranges_[idx] = (X[:, idx].min(), X[:, idx].max())
        
        # Create level hypervectors for continuous features
        self._seed_offset = 999999
        self.level_hvs_ = self._create_level_hvs(self.n_levels)
        
        # Encode training samples and create class prototypes
        for class_label in self.classes_:
            class_samples = X[y == class_label]
            class_hvs = []
            
            for sample in class_samples:
                encoded_hv = self._encode_sample(sample)
                class_hvs.append(encoded_hv)
            
            # Create class prototype by bundling all samples
            self.class_hvs_[class_label] = self._bundle(class_hvs)
        
        return self
    
    def _similarity(self, hv1, hv2):
        """Compute cosine similarity between two hypervectors."""
        return np.dot(hv1, hv2) / (self.n_dimensions)
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        check_is_fitted(self)
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        
        predictions = []
        
        for sample in X:
            # Encode test sample
            encoded_hv = self._encode_sample(sample)
            
            # Compute similarity with each class prototype
            similarities = {}
            for class_label, class_hv in self.class_hvs_.items():
                similarities[class_label] = self._similarity(encoded_hv, class_hv)
            
            # Predict class with highest similarity
            predicted_class = max(similarities, key=similarities.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self)
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        
        probabilities = []
        
        for sample in X:
            # Encode test sample
            encoded_hv = self._encode_sample(sample)
            
            # Compute similarity with each class prototype
            similarities = []
            for class_label in self.classes_:
                class_hv = self.class_hvs_[class_label]
                sim = self._similarity(encoded_hv, class_hv)
                similarities.append(sim)
            
            # Convert similarities to probabilities using softmax
            similarities = np.array(similarities)
            exp_sim = np.exp(similarities - np.max(similarities))
            proba = exp_sim / np.sum(exp_sim)
            probabilities.append(proba)
        
        return np.array(probabilities)