import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import zlib
from typing import Tuple, List


class BioInspiredCAClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that evolves Cellular Automata (CA) rules using bio-inspired
    metaheuristics (Genetic Algorithm). Fitness is measured by compression ratio
    of correctly classified training data.
    
    Parameters
    ----------
    n_rules : int, default=10
        Number of CA rules in the population
    n_generations : int, default=50
        Number of generations for evolution
    mutation_rate : float, default=0.1
        Probability of mutation for each bit
    crossover_rate : float, default=0.7
        Probability of crossover
    ca_steps : int, default=5
        Number of CA evolution steps
    elite_size : int, default=2
        Number of elite individuals to preserve
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_rules=10, n_generations=50, mutation_rate=0.1,
                 crossover_rate=0.7, ca_steps=5, elite_size=2, random_state=None):
        self.n_rules = n_rules
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.ca_steps = ca_steps
        self.elite_size = elite_size
        self.random_state = random_state
        
    def _initialize_population(self) -> np.ndarray:
        """Initialize population of CA rules (256 bits for elementary CA)."""
        return self.rng_.randint(0, 2, size=(self.n_rules, 256)).astype(np.uint8)
    
    def _apply_ca_rule(self, X: np.ndarray, rule: np.ndarray) -> np.ndarray:
        """
        Apply CA rule to input data for multiple steps.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        rule : array-like of shape (256,)
        
        Returns
        -------
        transformed : array-like of shape (n_samples, n_features)
        """
        # Binarize input based on median
        X_binary = (X > np.median(X, axis=0, keepdims=True)).astype(np.uint8)
        state = X_binary.copy()
        
        for _ in range(self.ca_steps):
            new_state = np.zeros_like(state)
            for i in range(state.shape[0]):
                for j in range(state.shape[1]):
                    # Get neighborhood (with periodic boundary)
                    left = state[i, (j - 1) % state.shape[1]]
                    center = state[i, j]
                    right = state[i, (j + 1) % state.shape[1]]
                    
                    # Calculate neighborhood index (3-bit number)
                    neighborhood = (left << 2) | (center << 1) | right
                    new_state[i, j] = rule[neighborhood]
            
            state = new_state
        
        return state
    
    def _extract_features(self, ca_output: np.ndarray) -> np.ndarray:
        """Extract features from CA output."""
        # Multiple statistical features per sample
        features = []
        features.append(np.sum(ca_output, axis=1))
        features.append(np.mean(ca_output, axis=1))
        features.append(np.std(ca_output, axis=1))
        features.append(np.max(ca_output, axis=1))
        features.append(np.min(ca_output, axis=1))
        
        # Add some pattern-based features
        # Count transitions (0->1 or 1->0)
        transitions = np.sum(np.abs(np.diff(ca_output, axis=1)), axis=1)
        features.append(transitions)
        
        return np.column_stack(features)
    
    def _classify(self, features: np.ndarray, y_train: np.ndarray, train_features: np.ndarray) -> np.ndarray:
        """Simple nearest centroid classification."""
        predictions = np.zeros(len(features), dtype=int)
        best_distances = np.full(len(features), np.inf)
        
        for class_label in self.classes_:
            class_mask = y_train == class_label
            if np.any(class_mask):
                # Calculate centroid for this class
                centroid = np.mean(train_features[class_mask], axis=0)
                
                # Calculate distances to centroid
                distances = np.linalg.norm(features - centroid, axis=1)
                
                # Update predictions where this class is closer
                mask = distances < best_distances
                predictions[mask] = class_label
                best_distances[mask] = distances[mask]
        
        return predictions
    
    def _compression_ratio(self, data: np.ndarray) -> float:
        """Calculate compression ratio using zlib."""
        if data.size == 0:
            return 1.0
        
        data_bytes = data.tobytes()
        if len(data_bytes) == 0:
            return 1.0
            
        try:
            compressed = zlib.compress(data_bytes, level=9)
            ratio = len(compressed) / max(len(data_bytes), 1)
            return ratio
        except:
            return 1.0
    
    def _fitness(self, rule: np.ndarray, X: np.ndarray, y: np.ndarray, 
                 train_features: np.ndarray = None) -> Tuple[float, np.ndarray]:
        """
        Calculate fitness based on compression ratio of correctly classified data.
        Lower compression ratio (better compression) = higher fitness.
        
        Returns
        -------
        fitness : float
        features : np.ndarray
        """
        # Apply CA rule
        ca_output = self._apply_ca_rule(X, rule)
        features = self._extract_features(ca_output)
        
        # For first evaluation, use features as train_features
        if train_features is None:
            train_features = features
        
        # Classify
        predictions = self._classify(features, y, train_features)
        
        # Get correctly classified samples
        correct_mask = predictions == y
        accuracy = np.mean(correct_mask)
        
        if accuracy == 0:
            return 0.0, features
        
        # Calculate compression ratio of correctly classified data
        correct_data = ca_output[correct_mask]
        
        if len(correct_data) == 0:
            return 0.0, features
        
        compression = self._compression_ratio(correct_data)
        
        # Fitness: balance accuracy and compression
        # Higher accuracy and lower compression ratio = higher fitness
        # Add small epsilon to avoid division by zero
        fitness = accuracy * (1.0 / (compression + 0.01))
        
        # Add diversity bonus based on feature variance
        feature_variance = np.mean(np.var(features, axis=0))
        fitness += 0.1 * feature_variance
        
        return fitness, features
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Two-point crossover."""
        if self.rng_.random() < self.crossover_rate:
            point1 = self.rng_.randint(1, len(parent1) - 1)
            point2 = self.rng_.randint(point1 + 1, len(parent1))
            
            child1 = parent1.copy()
            child2 = parent2.copy()
            
            child1[point1:point2] = parent2[point1:point2]
            child2[point1:point2] = parent1[point1:point2]
            
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Bit-flip mutation."""
        individual = individual.copy()
        mask = self.rng_.random(len(individual)) < self.mutation_rate
        individual[mask] = 1 - individual[mask]
        return individual
    
    def _select_parents(self, population: np.ndarray, fitness_scores: np.ndarray) -> np.ndarray:
        """Tournament selection."""
        tournament_size = 3
        selected = []
        
        for _ in range(len(population)):
            # Ensure we have enough individuals for tournament
            actual_tournament_size = min(tournament_size, len(population))
            tournament_idx = self.rng_.choice(len(population), size=actual_tournament_size, replace=False)
            tournament_fitness = fitness_scores[tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return np.array(selected)
    
    def fit(self, X, y):
        """
        Fit the classifier by evolving CA rules.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        self.rng_ = np.random.RandomState(self.random_state)
        
        # Initialize population
        population = self._initialize_population()
        
        best_fitness = -np.inf
        self.best_rule_ = None
        self.train_features_ = None
        
        # Evolution loop
        for generation in range(self.n_generations):
            # Evaluate fitness for all individuals
            fitness_scores = []
            all_features = []
            
            for rule in population:
                fitness, features = self._fitness(rule, X, y, self.train_features_)
                fitness_scores.append(fitness)
                all_features.append(features)
            
            fitness_scores = np.array(fitness_scores)
            
            # Track best
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                self.best_rule_ = population[gen_best_idx].copy()
                self.train_features_ = all_features[gen_best_idx]
            
            # Elitism: preserve best individuals
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elite = population[elite_indices].copy()
            
            # Selection
            parents = self._select_parents(population, fitness_scores)
            
            # Create new population through crossover and mutation
            new_population = []
            
            for i in range(0, len(parents) - 1, 2):
                child1, child2 = self._crossover(parents[i], parents[i + 1])
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.extend([child1, child2])
            
            # If odd number, add one more
            if len(new_population) < self.n_rules - self.elite_size:
                child = self._mutate(parents[-1].copy())
                new_population.append(child)
            
            # Ensure population size is maintained
            new_population = np.array(new_population[:self.n_rules - self.elite_size])
            population = np.vstack([elite, new_population])
        
        # Final training with best rule
        ca_output = self._apply_ca_rule(X, self.best_rule_)
        self.train_features_ = self._extract_features(ca_output)
        self.y_train_ = y
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        check_is_fitted(self, ['best_rule_', 'train_features_', 'y_train_'])
        X = check_array(X)
        
        # Apply best CA rule
        ca_output = self._apply_ca_rule(X, self.best_rule_)
        features = self._extract_features(ca_output)
        
        # Classify using stored training features
        predictions = self._classify(features, self.y_train_, self.train_features_)
        
        return predictions