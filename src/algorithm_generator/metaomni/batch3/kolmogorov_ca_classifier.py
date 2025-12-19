import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import zlib
from typing import Optional, Callable


class KolmogorovComplexityCAClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses Kolmogorov complexity approximations to prune
    cellular automaton (CA) rules, biasing toward simpler generalizable dynamics.
    
    This classifier evaluates CA rules based on their pattern complexity using
    compression-based approximations of Kolmogorov complexity. Rules generating
    high-complexity patterns are pruned, favoring simpler, more generalizable
    dynamics for classification.
    
    Parameters
    ----------
    n_rules : int, default=256
        Number of CA rules to evaluate (for elementary CA, max 256).
    
    n_steps : int, default=50
        Number of CA evolution steps to simulate.
    
    complexity_threshold : float, default=0.7
        Threshold for pruning high-complexity rules (0-1 scale).
        Rules with normalized complexity above this are pruned.
    
    min_rules : int, default=5
        Minimum number of rules to keep after pruning.
    
    random_state : int, optional
        Random seed for reproducibility.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes seen during fit.
    
    selected_rules_ : list
        CA rules selected after complexity-based pruning.
    
    rule_complexities_ : dict
        Complexity scores for each evaluated rule.
    
    class_prototypes_ : dict
        Prototype patterns for each class based on selected rules.
    """
    
    def __init__(
        self,
        n_rules: int = 256,
        n_steps: int = 50,
        complexity_threshold: float = 0.7,
        min_rules: int = 5,
        random_state: Optional[int] = None
    ):
        self.n_rules = n_rules
        self.n_steps = n_steps
        self.complexity_threshold = complexity_threshold
        self.min_rules = min_rules
        self.random_state = random_state
    
    def _kolmogorov_complexity_approx(self, data: np.ndarray) -> float:
        """
        Approximate Kolmogorov complexity using compression ratio.
        
        Parameters
        ----------
        data : np.ndarray
            Binary data to evaluate.
        
        Returns
        -------
        float
            Normalized complexity score (0-1).
        """
        # Convert to bytes
        data_bytes = np.packbits(data.astype(np.uint8)).tobytes()
        
        # Compress and calculate ratio
        compressed = zlib.compress(data_bytes, level=9)
        complexity = len(compressed) / max(len(data_bytes), 1)
        
        return min(complexity, 1.0)
    
    def _apply_ca_rule(self, rule_number: int, initial_state: np.ndarray, steps: int) -> np.ndarray:
        """
        Apply elementary CA rule to initial state for given steps.
        
        Parameters
        ----------
        rule_number : int
            CA rule number (0-255 for elementary CA).
        
        initial_state : np.ndarray
            Initial binary state.
        
        steps : int
            Number of evolution steps.
        
        Returns
        -------
        np.ndarray
            Evolution history of shape (steps, len(initial_state)).
        """
        size = len(initial_state)
        history = np.zeros((steps, size), dtype=np.uint8)
        history[0] = initial_state
        
        # Convert rule number to lookup table
        rule_binary = np.array([int(x) for x in format(rule_number, '08b')][::-1])
        
        for t in range(1, steps):
            for i in range(size):
                # Get neighborhood (with periodic boundary)
                left = history[t-1, (i-1) % size]
                center = history[t-1, i]
                right = history[t-1, (i+1) % size]
                
                # Compute neighborhood index
                neighborhood = left * 4 + center * 2 + right
                history[t, i] = rule_binary[neighborhood]
        
        return history
    
    def _evaluate_rule_complexity(self, rule_number: int, samples: np.ndarray) -> float:
        """
        Evaluate complexity of a CA rule across multiple samples.
        
        Parameters
        ----------
        rule_number : int
            CA rule number.
        
        samples : np.ndarray
            Sample initial states.
        
        Returns
        -------
        float
            Average complexity score.
        """
        complexities = []
        
        for sample in samples[:min(10, len(samples))]:  # Limit for efficiency
            # Binarize sample
            binary_sample = (sample > np.median(sample)).astype(np.uint8)
            
            # Apply CA rule
            evolution = self._apply_ca_rule(rule_number, binary_sample, self.n_steps)
            
            # Calculate complexity
            complexity = self._kolmogorov_complexity_approx(evolution.flatten())
            complexities.append(complexity)
        
        return np.mean(complexities)
    
    def _prune_rules(self, X: np.ndarray) -> list:
        """
        Prune CA rules based on complexity threshold.
        
        Parameters
        ----------
        X : np.ndarray
            Training data.
        
        Returns
        -------
        list
            Selected rule numbers.
        """
        rule_complexities = {}
        
        # Evaluate subset of rules
        rules_to_test = np.random.RandomState(self.random_state).choice(
            min(256, self.n_rules), 
            size=min(self.n_rules, 256), 
            replace=False
        )
        
        for rule in rules_to_test:
            complexity = self._evaluate_rule_complexity(rule, X)
            rule_complexities[rule] = complexity
        
        self.rule_complexities_ = rule_complexities
        
        # Sort by complexity (ascending - prefer simpler)
        sorted_rules = sorted(rule_complexities.items(), key=lambda x: x[1])
        
        # Select rules below threshold or minimum number
        selected = []
        for rule, complexity in sorted_rules:
            if complexity <= self.complexity_threshold or len(selected) < self.min_rules:
                selected.append(rule)
        
        return selected[:max(self.min_rules, len(selected))]
    
    def _extract_ca_features(self, X: np.ndarray, rules: list) -> np.ndarray:
        """
        Extract features by applying selected CA rules.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
        
        rules : list
            CA rules to apply.
        
        Returns
        -------
        np.ndarray
            Feature matrix.
        """
        features = []
        
        for sample in X:
            sample_features = []
            binary_sample = (sample > np.median(sample)).astype(np.uint8)
            
            for rule in rules:
                evolution = self._apply_ca_rule(rule, binary_sample, self.n_steps)
                
                # Extract statistical features from evolution
                sample_features.extend([
                    np.mean(evolution),
                    np.std(evolution),
                    self._kolmogorov_complexity_approx(evolution.flatten()),
                    np.sum(evolution[-1])  # Final state density
                ])
            
            features.append(sample_features)
        
        return np.array(features)
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier by selecting low-complexity CA rules.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        
        y_train : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self
            Fitted estimator.
        """
        X_train, y_train = check_X_y(X_train, y_train)
        self.classes_ = unique_labels(y_train)
        
        # Prune rules based on complexity
        self.selected_rules_ = self._prune_rules(X_train)
        
        # Extract features using selected rules
        X_features = self._extract_ca_features(X_train, self.selected_rules_)
        
        # Build class prototypes
        self.class_prototypes_ = {}
        for cls in self.classes_:
            mask = y_train == cls
            self.class_prototypes_[cls] = np.mean(X_features[mask], axis=0)
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ['selected_rules_', 'class_prototypes_'])
        X_test = check_array(X_test)
        
        # Extract features
        X_features = self._extract_ca_features(X_test, self.selected_rules_)
        
        # Predict based on nearest prototype
        predictions = []
        for features in X_features:
            distances = {}
            for cls, prototype in self.class_prototypes_.items():
                distances[cls] = np.linalg.norm(features - prototype)
            
            predictions.append(min(distances, key=distances.get))
        
        return np.array(predictions)