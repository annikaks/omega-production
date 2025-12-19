import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from collections import defaultdict
import zlib
from typing import List, Tuple, Dict


class AdaptiveContextMixingClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Context-Mixing Classifier using bio-inspired genetic algorithms.
    
    This classifier evolves compression dictionaries for each class using genetic
    algorithms, then uses compression-based distance for classification.
    
    Parameters
    ----------
    population_size : int, default=20
        Number of dictionaries in the genetic algorithm population per class.
    
    generations : int, default=10
        Number of evolutionary generations to run.
    
    mutation_rate : float, default=0.1
        Probability of mutation for each dictionary element.
    
    crossover_rate : float, default=0.7
        Probability of crossover between parent dictionaries.
    
    dictionary_size : int, default=256
        Maximum size of compression dictionary.
    
    elite_size : int, default=2
        Number of top dictionaries to preserve each generation.
    
    context_length : int, default=3
        Length of context patterns to extract.
    
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, population_size=20, generations=10, mutation_rate=0.1,
                 crossover_rate=0.7, dictionary_size=256, elite_size=2,
                 context_length=3, random_state=None):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.dictionary_size = dictionary_size
        self.elite_size = elite_size
        self.context_length = context_length
        self.random_state = random_state
    
    def _extract_contexts(self, X: np.ndarray) -> List[bytes]:
        """Extract context patterns from data."""
        contexts = []
        for sample in X:
            # Convert to bytes
            if sample.dtype != np.uint8:
                sample_normalized = ((sample - sample.min()) / 
                                   (sample.max() - sample.min() + 1e-10) * 255)
                sample_bytes = sample_normalized.astype(np.uint8).tobytes()
            else:
                sample_bytes = sample.tobytes()
            
            # Extract n-grams as contexts
            for i in range(len(sample_bytes) - self.context_length + 1):
                contexts.append(sample_bytes[i:i + self.context_length])
        
        return contexts
    
    def _create_dictionary(self, contexts: List[bytes]) -> bytes:
        """Create a compression dictionary from contexts."""
        # Count context frequencies
        context_freq = defaultdict(int)
        for ctx in contexts:
            context_freq[ctx] += 1
        
        # Select most frequent contexts up to dictionary_size
        sorted_contexts = sorted(context_freq.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
        
        dictionary = b''.join([ctx for ctx, _ in sorted_contexts[:self.dictionary_size]])
        return dictionary[:self.dictionary_size * self.context_length]
    
    def _initialize_population(self, X: np.ndarray) -> List[bytes]:
        """Initialize population of dictionaries."""
        population = []
        contexts = self._extract_contexts(X)
        
        for _ in range(self.population_size):
            # Sample random subset of contexts
            sample_size = min(len(contexts), self.dictionary_size)
            sampled_contexts = self.rng_.choice(len(contexts), 
                                               size=sample_size, 
                                               replace=False)
            sampled = [contexts[i] for i in sampled_contexts]
            dictionary = self._create_dictionary(sampled)
            population.append(dictionary)
        
        return population
    
    def _fitness(self, dictionary: bytes, X: np.ndarray) -> float:
        """Calculate fitness as compression ratio."""
        total_original = 0
        total_compressed = 0
        
        for sample in X:
            if sample.dtype != np.uint8:
                sample_normalized = ((sample - sample.min()) / 
                                   (sample.max() - sample.min() + 1e-10) * 255)
                sample_bytes = sample_normalized.astype(np.uint8).tobytes()
            else:
                sample_bytes = sample.tobytes()
            
            total_original += len(sample_bytes)
            
            # Compress with dictionary
            compressor = zlib.compressobj(zdict=dictionary)
            compressed = compressor.compress(sample_bytes)
            compressed += compressor.flush()
            total_compressed += len(compressed)
        
        # Higher fitness = better compression
        return total_original / (total_compressed + 1e-10)
    
    def _crossover(self, parent1: bytes, parent2: bytes) -> Tuple[bytes, bytes]:
        """Perform crossover between two parent dictionaries."""
        if self.rng_.random() > self.crossover_rate:
            return parent1, parent2
        
        # Single-point crossover
        point = self.rng_.randint(1, min(len(parent1), len(parent2)))
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        
        return child1, child2
    
    def _mutate(self, dictionary: bytes) -> bytes:
        """Mutate dictionary by randomly changing bytes."""
        dict_array = bytearray(dictionary)
        
        for i in range(len(dict_array)):
            if self.rng_.random() < self.mutation_rate:
                dict_array[i] = self.rng_.randint(0, 255)
        
        return bytes(dict_array)
    
    def _evolve_dictionary(self, X: np.ndarray) -> bytes:
        """Evolve optimal dictionary for given data using genetic algorithm."""
        population = self._initialize_population(X)
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [(dictionary, self._fitness(dictionary, X)) 
                            for dictionary in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Elitism: keep best dictionaries
            new_population = [dictionary for dictionary, _ in fitness_scores[:self.elite_size]]
            
            # Selection and reproduction
            while len(new_population) < self.population_size:
                # Tournament selection
                tournament_size = 3
                tournament = self.rng_.choice(len(fitness_scores), 
                                            size=tournament_size, 
                                            replace=False)
                parent1 = fitness_scores[tournament[0]][0]
                parent2 = fitness_scores[tournament[1]][0]
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # Return best dictionary
        final_fitness = [(dictionary, self._fitness(dictionary, X)) 
                        for dictionary in population]
        best_dictionary = max(final_fitness, key=lambda x: x[1])[0]
        
        return best_dictionary
    
    def _compression_distance(self, sample: np.ndarray, dictionary: bytes) -> float:
        """Calculate compression-based distance using dictionary."""
        if sample.dtype != np.uint8:
            sample_normalized = ((sample - sample.min()) / 
                               (sample.max() - sample.min() + 1e-10) * 255)
            sample_bytes = sample_normalized.astype(np.uint8).tobytes()
        else:
            sample_bytes = sample.tobytes()
        
        # Compress with dictionary
        compressor = zlib.compressobj(zdict=dictionary)
        compressed = compressor.compress(sample_bytes)
        compressed += compressor.flush()
        
        # Return normalized compression length
        return len(compressed) / (len(sample_bytes) + 1e-10)
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier by evolving optimal dictionaries for each class.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        
        y_train : array-like of shape (n_samples,)
            Target labels.
        
        Returns
        -------
        self : object
            Fitted classifier.
        """
        X_train, y_train = check_X_y(X_train, y_train)
        
        self.classes_ = unique_labels(y_train)
        self.n_features_in_ = X_train.shape[1]
        
        # Initialize random state
        self.rng_ = np.random.RandomState(self.random_state)
        
        # Evolve dictionary for each class
        self.class_dictionaries_ = {}
        
        for class_label in self.classes_:
            class_mask = y_train == class_label
            X_class = X_train[class_mask]
            
            # Evolve optimal dictionary for this class
            dictionary = self._evolve_dictionary(X_class)
            self.class_dictionaries_[class_label] = dictionary
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels using compression-based distance.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ['class_dictionaries_', 'classes_'])
        X_test = check_array(X_test)
        
        predictions = []
        
        for sample in X_test:
            # Calculate compression distance to each class dictionary
            distances = {}
            for class_label, dictionary in self.class_dictionaries_.items():
                distance = self._compression_distance(sample, dictionary)
                distances[class_label] = distance
            
            # Predict class with minimum compression distance
            predicted_class = min(distances.items(), key=lambda x: x[1])[0]
            predictions.append(predicted_class)
        
        return np.array(predictions)
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities based on compression distances.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ['class_dictionaries_', 'classes_'])
        X_test = check_array(X_test)
        
        probabilities = []
        
        for sample in X_test:
            # Calculate compression distance to each class dictionary
            distances = []
            for class_label in self.classes_:
                dictionary = self.class_dictionaries_[class_label]
                distance = self._compression_distance(sample, dictionary)
                distances.append(distance)
            
            # Convert distances to probabilities (inverse and normalize)
            distances = np.array(distances)
            inv_distances = 1.0 / (distances + 1e-10)
            proba = inv_distances / inv_distances.sum()
            probabilities.append(proba)
        
        return np.array(probabilities)