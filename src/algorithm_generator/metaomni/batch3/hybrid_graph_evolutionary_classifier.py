import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score
import networkx as nx
from scipy.spatial.distance import cdist
from typing import Optional, Callable


class HybridGraphEvolutionaryClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier using continuous-discrete hybrid encoding where strategy parameters
    evolve continuously but population structure follows discrete graph topology.
    
    Parameters
    ----------
    n_nodes : int, default=20
        Number of nodes in the population graph
    n_generations : int, default=100
        Number of evolutionary generations
    graph_type : str, default='small_world'
        Type of graph topology ('small_world', 'scale_free', 'ring', 'complete')
    mutation_rate : float, default=0.1
        Mutation rate for continuous parameters
    crossover_rate : float, default=0.7
        Crossover rate between neighboring nodes
    n_features_hidden : int, default=10
        Number of hidden features in the strategy
    learning_rate : float, default=0.01
        Learning rate for continuous parameter updates
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_nodes: int = 20,
        n_generations: int = 100,
        graph_type: str = 'small_world',
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        n_features_hidden: int = 10,
        learning_rate: float = 0.01,
        random_state: Optional[int] = None
    ):
        self.n_nodes = n_nodes
        self.n_generations = n_generations
        self.graph_type = graph_type
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_features_hidden = n_features_hidden
        self.learning_rate = learning_rate
        self.random_state = random_state
        
    def _create_graph(self) -> nx.Graph:
        """Create population graph based on specified topology."""
        if self.graph_type == 'small_world':
            G = nx.watts_strogatz_graph(self.n_nodes, k=4, p=0.3, seed=self.random_state)
        elif self.graph_type == 'scale_free':
            G = nx.barabasi_albert_graph(self.n_nodes, m=2, seed=self.random_state)
        elif self.graph_type == 'ring':
            G = nx.cycle_graph(self.n_nodes)
        elif self.graph_type == 'complete':
            G = nx.complete_graph(self.n_nodes)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
        return G
    
    def _initialize_population(self, n_features: int) -> dict:
        """Initialize population with continuous strategy parameters."""
        population = {}
        for node in range(self.n_nodes):
            # Each node has continuous parameters for a simple neural network
            population[node] = {
                'W1': self.rng_.normal(0, 0.1, (n_features, self.n_features_hidden)),
                'b1': np.zeros(self.n_features_hidden),
                'W2': self.rng_.normal(0, 0.1, (self.n_features_hidden, self.n_classes_)),
                'b2': np.zeros(self.n_classes_),
                'fitness': -np.inf
            }
        return population
    
    def _forward_pass(self, X: np.ndarray, params: dict) -> np.ndarray:
        """Forward pass through the neural network."""
        # Hidden layer with tanh activation
        hidden = np.tanh(X @ params['W1'] + params['b1'])
        # Output layer with softmax
        logits = hidden @ params['W2'] + params['b2']
        # Numerical stability for softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return probs
    
    def _evaluate_fitness(self, X: np.ndarray, y: np.ndarray, params: dict) -> float:
        """Evaluate fitness of a node's parameters."""
        probs = self._forward_pass(X, params)
        predictions = np.argmax(probs, axis=1)
        return accuracy_score(y, predictions)
    
    def _mutate(self, params: dict) -> dict:
        """Apply Gaussian mutation to continuous parameters."""
        mutated = {}
        for key in ['W1', 'b1', 'W2', 'b2']:
            if self.rng_.random() < self.mutation_rate:
                noise = self.rng_.normal(0, 0.01, params[key].shape)
                mutated[key] = params[key] + noise
            else:
                mutated[key] = params[key].copy()
        mutated['fitness'] = params['fitness']
        return mutated
    
    def _crossover(self, parent1: dict, parent2: dict) -> dict:
        """Perform crossover between two parent nodes."""
        child = {}
        for key in ['W1', 'b1', 'W2', 'b2']:
            if self.rng_.random() < self.crossover_rate:
                # Blend crossover
                alpha = self.rng_.random()
                child[key] = alpha * parent1[key] + (1 - alpha) * parent2[key]
            else:
                child[key] = parent1[key].copy() if self.rng_.random() < 0.5 else parent2[key].copy()
        child['fitness'] = -np.inf
        return child
    
    def _gradient_update(self, X: np.ndarray, y: np.ndarray, params: dict) -> dict:
        """Apply gradient-based update to continuous parameters."""
        # Forward pass
        hidden = np.tanh(X @ params['W1'] + params['b1'])
        logits = hidden @ params['W2'] + params['b2']
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # One-hot encode labels
        y_onehot = np.zeros((len(y), self.n_classes_))
        y_onehot[np.arange(len(y)), y] = 1
        
        # Backward pass
        d_logits = (probs - y_onehot) / len(y)
        d_W2 = hidden.T @ d_logits
        d_b2 = np.sum(d_logits, axis=0)
        
        d_hidden = d_logits @ params['W2'].T
        d_hidden_input = d_hidden * (1 - hidden ** 2)  # tanh derivative
        d_W1 = X.T @ d_hidden_input
        d_b1 = np.sum(d_hidden_input, axis=0)
        
        # Update parameters
        updated = {
            'W1': params['W1'] - self.learning_rate * d_W1,
            'b1': params['b1'] - self.learning_rate * d_b1,
            'W2': params['W2'] - self.learning_rate * d_W2,
            'b2': params['b2'] - self.learning_rate * d_b2,
            'fitness': params['fitness']
        }
        return updated
    
    def fit(self, X_train, y_train):
        """
        Fit the hybrid graph evolutionary classifier.
        
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
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_train.shape[1]
        
        # Initialize random state
        self.rng_ = np.random.RandomState(self.random_state)
        
        # Create discrete graph topology
        self.graph_ = self._create_graph()
        
        # Initialize population with continuous parameters
        self.population_ = self._initialize_population(self.n_features_in_)
        
        # Evolutionary loop
        for generation in range(self.n_generations):
            # Evaluate fitness for all nodes
            for node in self.population_:
                self.population_[node]['fitness'] = self._evaluate_fitness(
                    X_train, y_train, self.population_[node]
                )
            
            # Create new population
            new_population = {}
            
            for node in self.graph_.nodes():
                # Get neighbors in discrete graph
                neighbors = list(self.graph_.neighbors(node))
                
                if len(neighbors) > 0:
                    # Select best neighbor
                    neighbor_fitnesses = [self.population_[n]['fitness'] for n in neighbors]
                    best_neighbor = neighbors[np.argmax(neighbor_fitnesses)]
                    
                    # Crossover with best neighbor
                    if self.population_[best_neighbor]['fitness'] > self.population_[node]['fitness']:
                        offspring = self._crossover(
                            self.population_[node],
                            self.population_[best_neighbor]
                        )
                    else:
                        offspring = {k: v.copy() if isinstance(v, np.ndarray) else v 
                                   for k, v in self.population_[node].items()}
                else:
                    offspring = {k: v.copy() if isinstance(v, np.ndarray) else v 
                               for k, v in self.population_[node].items()}
                
                # Apply mutation
                offspring = self._mutate(offspring)
                
                # Apply gradient-based continuous update
                offspring = self._gradient_update(X_train, y_train, offspring)
                
                new_population[node] = offspring
            
            self.population_ = new_population
        
        # Select best individual as final model
        best_node = max(self.population_.keys(), 
                       key=lambda n: self.population_[n]['fitness'])
        self.best_params_ = self.population_[best_node]
        
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
        check_is_fitted(self, ['best_params_', 'classes_'])
        X_test = check_array(X_test)
        
        probs = self._forward_pass(X_test, self.best_params_)
        predictions = np.argmax(probs, axis=1)
        
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
            Predicted class probabilities
        """
        check_is_fitted(self, ['best_params_', 'classes_'])
        X_test = check_array(X_test)
        
        return self._forward_pass(X_test, self.best_params_)