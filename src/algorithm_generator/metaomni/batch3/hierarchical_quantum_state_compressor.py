import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
from scipy.linalg import expm
from typing import Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class HierarchicalQuantumStateCompressor(BaseEstimator, ClassifierMixin):
    """
    Hierarchical Quantum State Compression Classifier
    
    Implements a quantum-inspired hierarchical compression scheme where:
    - Lower-level qubits represent feature clusters
    - Higher-level qubits encode compressed representations based on mutual information
    - Classification is performed using quantum state overlap measurements
    
    Parameters
    ----------
    n_levels : int, default=3
        Number of hierarchical levels in the compression
    n_clusters_per_level : int, default=4
        Number of clusters at each level
    compression_ratio : float, default=0.5
        Ratio of compression between levels
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_levels: int = 3,
        n_clusters_per_level: int = 4,
        compression_ratio: float = 0.5,
        random_state: Optional[int] = 42
    ):
        self.n_levels = n_levels
        self.n_clusters_per_level = n_clusters_per_level
        self.compression_ratio = compression_ratio
        self.random_state = random_state
        
    def _initialize_quantum_state(self, dim: int) -> np.ndarray:
        """Initialize a quantum state vector in computational basis."""
        state = np.zeros(dim, dtype=complex)
        state[0] = 1.0
        return state
    
    def _create_rotation_operator(self, theta: float, phi: float, dim: int) -> np.ndarray:
        """Create a rotation operator for quantum state manipulation."""
        # Generalized rotation in Hilbert space
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(min(2, dim)):
            for j in range(min(2, dim)):
                if i == j:
                    H[i, j] = np.cos(theta) * np.exp(1j * phi * i)
                else:
                    H[i, j] = np.sin(theta) * np.exp(1j * phi * (i + j))
        return expm(-1j * H)
    
    def _encode_features_to_quantum_state(self, features: np.ndarray) -> np.ndarray:
        """Encode classical features into quantum state amplitudes."""
        # Normalize features to create valid quantum state
        norm = np.linalg.norm(features)
        if norm > 0:
            amplitudes = features / norm
        else:
            amplitudes = features
        
        # Ensure complex amplitudes
        state = amplitudes.astype(complex)
        
        # Renormalize to ensure valid quantum state
        norm = np.linalg.norm(state)
        if norm > 0:
            state = state / norm
        else:
            state = self._initialize_quantum_state(len(features))
            
        return state
    
    def _compute_mutual_information(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute mutual information between features and labels."""
        n_features = features.shape[1]
        mi_scores = np.zeros(n_features)
        
        for i in range(n_features):
            # Discretize continuous features for MI calculation
            feature_discrete = np.digitize(features[:, i], bins=np.linspace(
                features[:, i].min(), features[:, i].max(), 10
            ))
            mi_scores[i] = mutual_info_score(labels, feature_discrete)
        
        return mi_scores
    
    def _cluster_features(self, X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, KMeans]:
        """Cluster features based on their patterns."""
        if X.shape[1] < n_clusters:
            n_clusters = max(1, X.shape[1])
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        
        # Cluster based on feature correlations
        feature_correlations = np.corrcoef(X.T)
        
        if feature_correlations.shape[0] >= n_clusters:
            cluster_labels = kmeans.fit_predict(feature_correlations)
        else:
            cluster_labels = np.arange(feature_correlations.shape[0])
            
        return cluster_labels, kmeans
    
    def _compress_level(
        self,
        states: List[np.ndarray],
        mi_scores: np.ndarray,
        level: int
    ) -> List[np.ndarray]:
        """Compress quantum states at a given level based on mutual information."""
        n_output = max(1, int(len(states) * self.compression_ratio))
        
        # Sort states by mutual information scores
        if len(mi_scores) == len(states):
            sorted_indices = np.argsort(mi_scores)[::-1]
        else:
            sorted_indices = np.arange(len(states))
        
        compressed_states = []
        
        for i in range(n_output):
            # Select states to compress together
            group_size = len(states) // n_output
            start_idx = i * group_size
            end_idx = start_idx + group_size if i < n_output - 1 else len(states)
            
            if start_idx >= len(sorted_indices):
                break
                
            group_indices = sorted_indices[start_idx:end_idx]
            
            # Combine states through quantum superposition
            combined_state = np.zeros_like(states[0])
            for idx in group_indices:
                if idx < len(states):
                    combined_state += states[idx]
            
            # Normalize
            norm = np.linalg.norm(combined_state)
            if norm > 0:
                combined_state = combined_state / norm
            
            compressed_states.append(combined_state)
        
        return compressed_states
    
    def _build_hierarchy(self, X: np.ndarray, y: np.ndarray) -> List[List[np.ndarray]]:
        """Build hierarchical quantum state representation."""
        hierarchy = []
        
        # Level 0: Encode individual features
        level_0_states = []
        for i in range(X.shape[1]):
            feature_col = X[:, i].reshape(-1, 1)
            # Create quantum state from feature statistics
            feature_stats = np.array([
                np.mean(feature_col),
                np.std(feature_col),
                np.min(feature_col),
                np.max(feature_col)
            ])
            state = self._encode_features_to_quantum_state(feature_stats)
            level_0_states.append(state)
        
        hierarchy.append(level_0_states)
        
        # Compute mutual information for compression guidance
        mi_scores = self._compute_mutual_information(X, y)
        
        # Build higher levels through compression
        current_states = level_0_states
        current_mi = mi_scores
        
        for level in range(1, self.n_levels):
            compressed_states = self._compress_level(current_states, current_mi, level)
            hierarchy.append(compressed_states)
            
            # Update MI scores for next level
            if len(current_mi) > len(compressed_states):
                # Aggregate MI scores
                group_size = len(current_mi) // len(compressed_states)
                new_mi = []
                for i in range(len(compressed_states)):
                    start_idx = i * group_size
                    end_idx = start_idx + group_size if i < len(compressed_states) - 1 else len(current_mi)
                    new_mi.append(np.mean(current_mi[start_idx:end_idx]))
                current_mi = np.array(new_mi)
            
            current_states = compressed_states
        
        return hierarchy
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the hierarchical quantum state compressor.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target labels
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        # Store classes
        self.classes_ = np.unique(y_train)
        
        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_train)
        
        # Build hierarchical quantum representations for each class
        self.class_hierarchies_ = {}
        
        for cls in self.classes_:
            cls_mask = y_train == cls
            X_cls = X_scaled[cls_mask]
            y_cls = y_train[cls_mask]
            
            # Build hierarchy for this class
            hierarchy = self._build_hierarchy(X_cls, y_cls)
            self.class_hierarchies_[cls] = hierarchy
        
        return self
    
    def _compute_state_fidelity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Compute quantum state fidelity (overlap)."""
        # Ensure same dimension
        min_dim = min(len(state1), len(state2))
        state1 = state1[:min_dim]
        state2 = state2[:min_dim]
        
        # Compute fidelity as |<state1|state2>|^2
        overlap = np.abs(np.vdot(state1, state2))
        fidelity = overlap ** 2
        
        return fidelity
    
    def _compute_hierarchy_similarity(
        self,
        test_hierarchy: List[List[np.ndarray]],
        class_hierarchy: List[List[np.ndarray]]
    ) -> float:
        """Compute similarity between two hierarchies."""
        total_similarity = 0.0
        total_weight = 0.0
        
        # Weight higher levels more (they contain compressed information)
        for level in range(min(len(test_hierarchy), len(class_hierarchy))):
            level_weight = (level + 1) ** 2  # Quadratic weighting
            
            test_states = test_hierarchy[level]
            class_states = class_hierarchy[level]
            
            level_similarity = 0.0
            n_comparisons = 0
            
            for test_state in test_states:
                for class_state in class_states:
                    fidelity = self._compute_state_fidelity(test_state, class_state)
                    level_similarity += fidelity
                    n_comparisons += 1
            
            if n_comparisons > 0:
                level_similarity /= n_comparisons
                total_similarity += level_weight * level_similarity
                total_weight += level_weight
        
        if total_weight > 0:
            return total_similarity / total_weight
        return 0.0
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for test samples.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        X_test = np.asarray(X_test)
        X_scaled = self.scaler_.transform(X_test)
        
        predictions = []
        
        for i in range(X_test.shape[0]):
            sample = X_scaled[i:i+1]
            
            # Create dummy labels for hierarchy building
            dummy_labels = np.zeros(1)
            
            # Build hierarchy for test sample
            test_hierarchy = self._build_hierarchy(sample, dummy_labels)
            
            # Compare with each class hierarchy
            best_class = self.classes_[0]
            best_similarity = -1.0
            
            for cls in self.classes_:
                class_hierarchy = self.class_hierarchies_[cls]
                similarity = self._compute_hierarchy_similarity(test_hierarchy, class_hierarchy)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_class = cls
            
            predictions.append(best_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for test samples.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        X_test = np.asarray(X_test)
        X_scaled = self.scaler_.transform(X_test)
        
        probabilities = []
        
        for i in range(X_test.shape[0]):
            sample = X_scaled[i:i+1]
            dummy_labels = np.zeros(1)
            test_hierarchy = self._build_hierarchy(sample, dummy_labels)
            
            similarities = []
            for cls in self.classes_:
                class_hierarchy = self.class_hierarchies_[cls]
                similarity = self._compute_hierarchy_similarity(test_hierarchy, class_hierarchy)
                similarities.append(similarity)
            
            # Convert similarities to probabilities
            similarities = np.array(similarities)
            if np.sum(similarities) > 0:
                probs = similarities / np.sum(similarities)
            else:
                probs = np.ones(len(self.classes_)) / len(self.classes_)
            
            probabilities.append(probs)
        
        return np.array(probabilities)