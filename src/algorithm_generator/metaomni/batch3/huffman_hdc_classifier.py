import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import heapq


class HuffmanHDCClassifier:
    """
    Hyperdimensional Computing classifier with progressive compression using
    Huffman-inspired encoding for frequently co-occurring patterns.
    
    This classifier encodes patterns in high-dimensional space where frequent
    patterns get shorter (sparser) representations, similar to Huffman coding.
    """
    
    def __init__(
        self,
        n_dimensions: int = 10000,
        n_levels: int = 256,
        compression_threshold: float = 0.1,
        min_pattern_freq: int = 2,
        random_state: Optional[int] = None
    ):
        """
        Initialize the Huffman HDC Classifier.
        
        Parameters:
        -----------
        n_dimensions : int
            Dimensionality of hypervectors
        n_levels : int
            Number of quantization levels for continuous features
        compression_threshold : float
            Frequency threshold for pattern compression (0-1)
        min_pattern_freq : int
            Minimum frequency for a pattern to be considered for compression
        random_state : int or None
            Random seed for reproducibility
        """
        self.n_dimensions = n_dimensions
        self.n_levels = n_levels
        self.compression_threshold = compression_threshold
        self.min_pattern_freq = min_pattern_freq
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # Storage for learned representations
        self.feature_hvs: Dict[int, np.ndarray] = {}
        self.level_hvs: Dict[int, np.ndarray] = {}
        self.position_hvs: Dict[int, np.ndarray] = {}
        self.class_hvs: Dict = {}
        
        # Pattern compression structures
        self.pattern_frequencies: Counter = Counter()
        self.compressed_patterns: Dict[Tuple, np.ndarray] = {}
        self.compression_map: Dict[Tuple, int] = {}
        
    def _generate_random_hv(self, sparsity: float = 0.5) -> np.ndarray:
        """Generate a random bipolar hypervector with given sparsity."""
        hv = np.zeros(self.n_dimensions)
        n_active = int(self.n_dimensions * sparsity)
        indices = self.rng.choice(self.n_dimensions, n_active, replace=False)
        hv[indices] = self.rng.choice([-1, 1], n_active)
        return hv
    
    def _bind(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """Bind two hypervectors (element-wise multiplication)."""
        return hv1 * hv2
    
    def _bundle(self, hvs: List[np.ndarray]) -> np.ndarray:
        """Bundle multiple hypervectors (element-wise sum + threshold)."""
        if not hvs:
            return np.zeros(self.n_dimensions)
        summed = np.sum(hvs, axis=0)
        return np.sign(summed)
    
    def _permute(self, hv: np.ndarray, shift: int = 1) -> np.ndarray:
        """Permute hypervector by rotation."""
        return np.roll(hv, shift)
    
    def _similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute cosine similarity between two hypervectors."""
        norm1 = np.linalg.norm(hv1)
        norm2 = np.linalg.norm(hv2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(hv1, hv2) / (norm1 * norm2)
    
    def _extract_patterns(self, X: np.ndarray) -> List[List[Tuple]]:
        """Extract co-occurring patterns from data."""
        patterns_per_sample = []
        
        for sample in X:
            sample_patterns = []
            # Extract bigrams and trigrams as patterns
            for i in range(len(sample) - 1):
                bigram = (i, i+1, tuple(sample[i:i+2]))
                sample_patterns.append(bigram)
            
            for i in range(len(sample) - 2):
                trigram = (i, i+1, i+2, tuple(sample[i:i+3]))
                sample_patterns.append(trigram)
            
            patterns_per_sample.append(sample_patterns)
        
        return patterns_per_sample
    
    def _build_huffman_tree(self, frequencies: Counter) -> Dict[Tuple, int]:
        """Build Huffman-inspired compression levels based on frequencies."""
        if not frequencies:
            return {}
        
        # Create priority queue with (frequency, unique_id, pattern)
        heap = [(freq, idx, pattern) for idx, (pattern, freq) in enumerate(frequencies.items())]
        heapq.heapify(heap)
        
        # Assign compression levels (lower level = more compression)
        compression_levels = {}
        total_patterns = len(frequencies)
        
        for rank, (freq, _, pattern) in enumerate(sorted(heap, reverse=True)):
            # More frequent patterns get lower compression levels (higher sparsity)
            compression_level = int((rank / total_patterns) * 10)
            compression_levels[pattern] = compression_level
        
        return compression_levels
    
    def _create_compressed_hv(self, pattern: Tuple, compression_level: int) -> np.ndarray:
        """Create a compressed hypervector based on compression level."""
        # Higher compression level = higher sparsity (shorter representation)
        sparsity = 0.1 + (compression_level / 10) * 0.4  # Range: 0.1 to 0.5
        return self._generate_random_hv(sparsity=sparsity)
    
    def _encode_sample(self, sample: np.ndarray, use_compression: bool = True) -> np.ndarray:
        """Encode a single sample into a hypervector."""
        feature_hvs = []
        
        for pos, value in enumerate(sample):
            # Get or create feature and position hypervectors
            if pos not in self.feature_hvs:
                self.feature_hvs[pos] = self._generate_random_hv()
            if pos not in self.position_hvs:
                self.position_hvs[pos] = self._generate_random_hv()
            
            # Quantize continuous values
            level = int(np.clip(value * self.n_levels, 0, self.n_levels - 1))
            if level not in self.level_hvs:
                self.level_hvs[level] = self._generate_random_hv()
            
            # Bind feature, position, and level
            feature_hv = self._bind(
                self.feature_hvs[pos],
                self._bind(self.position_hvs[pos], self.level_hvs[level])
            )
            feature_hvs.append(feature_hv)
        
        # Apply compression for frequent patterns if enabled
        if use_compression and self.compressed_patterns:
            for i in range(len(sample) - 1):
                bigram_key = (i, i+1, tuple(sample[i:i+2]))
                if bigram_key in self.compressed_patterns:
                    # Replace with compressed representation
                    compressed_hv = self.compressed_patterns[bigram_key]
                    feature_hvs.append(compressed_hv)
        
        return self._bundle(feature_hvs)
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the classifier to training data.
        
        Parameters:
        -----------
        X_train : np.ndarray of shape (n_samples, n_features)
            Training data
        y_train : np.ndarray of shape (n_samples,)
            Training labels
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        # Normalize features to [0, 1]
        self.feature_min_ = X_train.min(axis=0)
        self.feature_max_ = X_train.max(axis=0)
        feature_range = self.feature_max_ - self.feature_min_
        feature_range[feature_range == 0] = 1  # Avoid division by zero
        X_normalized = (X_train - self.feature_min_) / feature_range
        
        # Extract and count patterns
        all_patterns = self._extract_patterns(X_normalized)
        for patterns in all_patterns:
            self.pattern_frequencies.update(patterns)
        
        # Build compression map for frequent patterns
        frequent_patterns = {
            pattern: freq for pattern, freq in self.pattern_frequencies.items()
            if freq >= self.min_pattern_freq
        }
        
        if frequent_patterns:
            self.compression_map = self._build_huffman_tree(Counter(frequent_patterns))
            
            # Create compressed hypervectors for frequent patterns
            for pattern, compression_level in self.compression_map.items():
                self.compressed_patterns[pattern] = self._create_compressed_hv(
                    pattern, compression_level
                )
        
        # Encode training samples and create class prototypes
        self.classes_ = np.unique(y_train)
        class_hvs_lists = defaultdict(list)
        
        for sample, label in zip(X_normalized, y_train):
            sample_hv = self._encode_sample(sample, use_compression=True)
            class_hvs_lists[label].append(sample_hv)
        
        # Create class prototypes by bundling all samples of each class
        for label in self.classes_:
            self.class_hvs[label] = self._bundle(class_hvs_lists[label])
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for test data.
        
        Parameters:
        -----------
        X_test : np.ndarray of shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted labels
        """
        X_test = np.asarray(X_test)
        
        # Normalize using training statistics
        feature_range = self.feature_max_ - self.feature_min_
        feature_range[feature_range == 0] = 1
        X_normalized = (X_test - self.feature_min_) / feature_range
        X_normalized = np.clip(X_normalized, 0, 1)
        
        predictions = []
        for sample in X_normalized:
            sample_hv = self._encode_sample(sample, use_compression=True)
            
            # Find most similar class prototype
            best_similarity = -np.inf
            best_class = self.classes_[0]
            
            for label in self.classes_:
                similarity = self._similarity(sample_hv, self.class_hvs[label])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_class = label
            
            predictions.append(best_class)
        
        return np.array(predictions)
    
    def get_compression_stats(self) -> Dict:
        """Return statistics about pattern compression."""
        return {
            'total_patterns': len(self.pattern_frequencies),
            'compressed_patterns': len(self.compressed_patterns),
            'compression_ratio': len(self.compressed_patterns) / max(len(self.pattern_frequencies), 1),
            'most_frequent_patterns': self.pattern_frequencies.most_common(10)
        }