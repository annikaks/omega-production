import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
from typing import Tuple, List
import warnings


class GeneticCodebookClassifier(BaseEstimator, ClassifierMixin):
    """
    A bio-inspired genetic algorithm classifier that evolves compression codebooks.
    
    The classifier evolves populations of codebooks that compress input features while
    maintaining predictive accuracy. Fitness is evaluated based on both compactness
    (codebook size) and classification accuracy.
    
    Parameters
    ----------
    codebook_size : int, default=10
        Maximum number of prototype vectors in each codebook.
    population_size : int, default=50
        Number of individuals in the genetic algorithm population.
    n_generations : int, default=100
        Number of generations to evolve.
    mutation_rate : float, default=0.1
        Probability of mutation for each gene.
    crossover_rate : float, default=0.7
        Probability of crossover between parents.
    compactness_weight : float, default=0.3
        Weight for codebook compactness in fitness (0-1).
    elitism_rate : float, default=0.1
        Fraction of top individuals to preserve each generation.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        codebook_size: int = 10,
        population_size: int = 50,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        compactness_weight: float = 0.3,
        elitism_rate: float = 0.1,
        random_state: int = None
    ):
        self.codebook_size = codebook_size
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.compactness_weight = compactness_weight
        self.elitism_rate = elitism_rate
        self.random_state = random_state
        
    def _initialize_population(self, X: np.ndarray, y: np.ndarray) -> List[dict]:
        """Initialize population with random codebooks."""
        population = []
        
        for _ in range(self.population_size):
            # Randomly select prototype indices from training data
            # Use randint instead of integers for compatibility
            n_prototypes = self.rng_.randint(2, self.codebook_size + 1)
            prototype_indices = self.rng_.choice(
                len(X), size=n_prototypes, replace=False
            )
            
            individual = {
                'prototypes': X[prototype_indices].copy(),
                'labels': y[prototype_indices].copy(),
                'fitness': 0.0
            }
            population.append(individual)
            
        return population
    
    def _encode_sample(self, x: np.ndarray, prototypes: np.ndarray) -> int:
        """Encode a sample by finding nearest prototype."""
        distances = np.linalg.norm(prototypes - x, axis=1)
        return np.argmin(distances)
    
    def _predict_with_codebook(
        self, X: np.ndarray, prototypes: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Predict labels using a codebook."""
        predictions = np.zeros(len(X), dtype=labels.dtype)
        
        for i, x in enumerate(X):
            nearest_idx = self._encode_sample(x, prototypes)
            predictions[i] = labels[nearest_idx]
            
        return predictions
    
    def _evaluate_fitness(
        self, individual: dict, X: np.ndarray, y: np.ndarray
    ) -> float:
        """
        Evaluate fitness based on accuracy and compactness.
        
        Fitness = (1 - compactness_weight) * accuracy - compactness_weight * size_penalty
        """
        prototypes = individual['prototypes']
        labels = individual['labels']
        
        # Predictive accuracy
        predictions = self._predict_with_codebook(X, prototypes, labels)
        accuracy = accuracy_score(y, predictions)
        
        # Compactness (normalized by max codebook size)
        size_penalty = len(prototypes) / self.codebook_size
        
        # Combined fitness
        fitness = (1 - self.compactness_weight) * accuracy - \
                  self.compactness_weight * size_penalty
        
        return fitness
    
    def _tournament_selection(
        self, population: List[dict], tournament_size: int = 3
    ) -> dict:
        """Select individual using tournament selection."""
        # Ensure tournament size doesn't exceed population size
        tournament_size = min(tournament_size, len(population))
        tournament_indices = self.rng_.choice(
            len(population), size=tournament_size, replace=False
        )
        tournament = [population[i] for i in tournament_indices]
        return max(tournament, key=lambda ind: ind['fitness'])
    
    def _crossover(self, parent1: dict, parent2: dict) -> Tuple[dict, dict]:
        """Perform crossover between two parents."""
        if self.rng_.random() > self.crossover_rate:
            return (
                {'prototypes': parent1['prototypes'].copy(), 
                 'labels': parent1['labels'].copy(), 'fitness': 0.0},
                {'prototypes': parent2['prototypes'].copy(), 
                 'labels': parent2['labels'].copy(), 'fitness': 0.0}
            )
        
        # Uniform crossover: randomly select prototypes from both parents
        all_prototypes = np.vstack([parent1['prototypes'], parent2['prototypes']])
        all_labels = np.concatenate([parent1['labels'], parent2['labels']])
        
        # Remove duplicates and limit size
        unique_indices = []
        seen = set()
        for i, proto in enumerate(all_prototypes):
            proto_tuple = tuple(proto)
            if proto_tuple not in seen:
                seen.add(proto_tuple)
                unique_indices.append(i)
        
        unique_indices = np.array(unique_indices)
        
        if len(unique_indices) > self.codebook_size:
            selected = self.rng_.choice(
                unique_indices, size=self.codebook_size, replace=False
            )
        else:
            selected = unique_indices
        
        # Ensure we have at least 2 prototypes for each offspring
        if len(selected) < 2:
            # If we have less than 2, just return copies of parents
            return (
                {'prototypes': parent1['prototypes'].copy(), 
                 'labels': parent1['labels'].copy(), 'fitness': 0.0},
                {'prototypes': parent2['prototypes'].copy(), 
                 'labels': parent2['labels'].copy(), 'fitness': 0.0}
            )
        
        # Split for two offspring
        split_point = max(1, len(selected) // 2)
        
        indices1 = selected[:split_point]
        indices2 = selected[split_point:]
        
        # Ensure both offspring have at least one prototype
        if len(indices2) == 0:
            indices2 = selected[-1:]
            indices1 = selected[:-1]
            if len(indices1) == 0:
                indices1 = selected[:1]
        
        offspring1 = {
            'prototypes': all_prototypes[indices1].copy(),
            'labels': all_labels[indices1].copy(),
            'fitness': 0.0
        }
        
        offspring2 = {
            'prototypes': all_prototypes[indices2].copy(),
            'labels': all_labels[indices2].copy(),
            'fitness': 0.0
        }
        
        return offspring1, offspring2
    
    def _mutate(self, individual: dict, X: np.ndarray, y: np.ndarray) -> dict:
        """Mutate an individual."""
        prototypes = individual['prototypes'].copy()
        labels = individual['labels'].copy()
        
        # Mutation: replace, add, or remove prototypes
        if self.rng_.random() < self.mutation_rate:
            mutation_type = self.rng_.choice(['replace', 'add', 'remove'])
            
            if mutation_type == 'replace' and len(prototypes) > 0:
                # Replace a random prototype
                idx = self.rng_.randint(0, len(prototypes))
                new_idx = self.rng_.randint(0, len(X))
                prototypes[idx] = X[new_idx]
                labels[idx] = y[new_idx]
                
            elif mutation_type == 'add' and len(prototypes) < self.codebook_size:
                # Add a new prototype
                new_idx = self.rng_.randint(0, len(X))
                prototypes = np.vstack([prototypes, X[new_idx:new_idx+1]])
                labels = np.concatenate([labels, y[new_idx:new_idx+1]])
                
            elif mutation_type == 'remove' and len(prototypes) > 2:
                # Remove a random prototype (keep at least 2)
                idx = self.rng_.randint(0, len(prototypes))
                prototypes = np.delete(prototypes, idx, axis=0)
                labels = np.delete(labels, idx)
        
        # Gaussian noise mutation on prototype values
        if self.rng_.random() < self.mutation_rate:
            noise = self.rng_.normal(0, 0.1, prototypes.shape)
            prototypes = prototypes + noise
        
        return {
            'prototypes': prototypes,
            'labels': labels,
            'fitness': 0.0
        }
    
    def fit(self, X, y):
        """
        Fit the genetic codebook classifier.
        
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
        self.n_features_in_ = X.shape[1]
        
        # Initialize random state
        self.rng_ = np.random.RandomState(self.random_state)
        
        # Ensure codebook size is reasonable
        if self.codebook_size > len(X):
            warnings.warn(
                f"codebook_size ({self.codebook_size}) is larger than number of "
                f"samples ({len(X)}). Setting codebook_size to {len(X)}."
            )
            self.codebook_size = len(X)
        
        # Initialize population
        population = self._initialize_population(X, y)
        
        # Evaluate initial population
        for individual in population:
            individual['fitness'] = self._evaluate_fitness(individual, X, y)
        
        # Evolution loop
        best_fitness_history = []
        
        for generation in range(self.n_generations):
            # Sort by fitness
            population.sort(key=lambda ind: ind['fitness'], reverse=True)
            best_fitness_history.append(population[0]['fitness'])
            
            # Elitism: preserve top individuals
            n_elite = max(1, int(self.elitism_rate * self.population_size))
            new_population = [
                {
                    'prototypes': ind['prototypes'].copy(),
                    'labels': ind['labels'].copy(),
                    'fitness': ind['fitness']
                }
                for ind in population[:n_elite]
            ]
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # Crossover
                offspring1, offspring2 = self._crossover(parent1, parent2)
                
                # Mutation
                offspring1 = self._mutate(offspring1, X, y)
                offspring2 = self._mutate(offspring2, X, y)
                
                # Evaluate fitness
                offspring1['fitness'] = self._evaluate_fitness(offspring1, X, y)
                offspring2['fitness'] = self._evaluate_fitness(offspring2, X, y)
                
                new_population.extend([offspring1, offspring2])
            
            # Trim to population size
            population = new_population[:self.population_size]
        
        # Select best individual
        population.sort(key=lambda ind: ind['fitness'], reverse=True)
        self.best_individual_ = population[0]
        self.fitness_history_ = best_fitness_history
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fitted
        check_is_fitted(self, ['best_individual_', 'classes_'])
        
        # Validate input
        X = check_array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but classifier expects "
                f"{self.n_features_in_} features."
            )
        
        # Predict using best codebook
        predictions = self._predict_with_codebook(
            X,
            self.best_individual_['prototypes'],
            self.best_individual_['labels']
        )
        
        return predictions
    
    def get_codebook(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the evolved codebook (prototypes and labels).
        
        Returns
        -------
        prototypes : ndarray of shape (n_prototypes, n_features)
            Prototype vectors.
        labels : ndarray of shape (n_prototypes,)
            Labels for each prototype.
        """
        check_is_fitted(self, ['best_individual_'])
        return (
            self.best_individual_['prototypes'].copy(),
            self.best_individual_['labels'].copy()
        )
    
    def get_fitness_history(self) -> np.ndarray:
        """
        Get the fitness history across generations.
        
        Returns
        -------
        fitness_history : ndarray of shape (n_generations,)
            Best fitness value for each generation.
        """
        check_is_fitted(self, ['fitness_history_'])
        return np.array(self.fitness_history_)