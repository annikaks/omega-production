import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class HyperdimensionalPSOClassifier(BaseEstimator, ClassifierMixin):
    """
    Particle Swarm Optimization classifier in hyperdimensional space.
    
    Each particle represents a candidate class hypervector that evolves toward
    optimal separability using PSO dynamics.
    
    Parameters
    ----------
    n_dimensions : int, default=10000
        Dimensionality of the hypervector space.
    n_particles : int, default=30
        Number of particles per class in the swarm.
    n_iterations : int, default=100
        Number of PSO iterations for optimization.
    w : float, default=0.7
        Inertia weight for velocity update.
    c1 : float, default=1.5
        Cognitive parameter (personal best attraction).
    c2 : float, default=1.5
        Social parameter (global best attraction).
    random_state : int, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    class_hypervectors_ : dict
        Optimized hypervector for each class.
    """
    
    def __init__(self, n_dimensions=10000, n_particles=30, n_iterations=100,
                 w=0.7, c1=1.5, c2=1.5, random_state=None):
        self.n_dimensions = n_dimensions
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state
    
    def _encode_sample(self, x):
        """Encode a sample into hyperdimensional space using random projection."""
        return np.sign(self.projection_matrix_ @ x + self.bias_)
    
    def _cosine_similarity(self, v1, v2):
        """Compute cosine similarity between two hypervectors."""
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    
    def _fitness(self, hypervector, class_label, X_encoded, y):
        """
        Compute fitness of a candidate hypervector.
        
        Fitness is based on:
        - High similarity to samples of the same class
        - Low similarity to samples of other classes
        """
        same_class_mask = (y == class_label)
        other_class_mask = ~same_class_mask
        
        same_class_similarities = []
        for sample in X_encoded[same_class_mask]:
            same_class_similarities.append(self._cosine_similarity(hypervector, sample))
        
        other_class_similarities = []
        for sample in X_encoded[other_class_mask]:
            other_class_similarities.append(self._cosine_similarity(hypervector, sample))
        
        # Fitness: maximize intra-class similarity, minimize inter-class similarity
        intra_class_score = np.mean(same_class_similarities) if same_class_similarities else 0
        inter_class_score = np.mean(other_class_similarities) if other_class_similarities else 0
        
        # Add margin-based separation
        fitness = intra_class_score - inter_class_score
        
        return fitness
    
    def _optimize_class_hypervector(self, class_label, X_encoded, y):
        """
        Use PSO to find optimal hypervector for a given class.
        """
        rng = np.random.RandomState(self.random_state)
        
        # Initialize particles (positions in hyperdimensional space)
        particles = rng.randn(self.n_particles, self.n_dimensions)
        particles = particles / (np.linalg.norm(particles, axis=1, keepdims=True) + 1e-10)
        
        # Initialize velocities
        velocities = rng.randn(self.n_particles, self.n_dimensions) * 0.1
        
        # Initialize personal best positions and fitness
        personal_best_positions = particles.copy()
        personal_best_fitness = np.array([
            self._fitness(p, class_label, X_encoded, y) for p in particles
        ])
        
        # Initialize global best
        global_best_idx = np.argmax(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        # PSO iterations
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # Update velocity
                r1, r2 = rng.rand(2)
                cognitive = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                social = self.c2 * r2 * (global_best_position - particles[i])
                velocities[i] = self.w * velocities[i] + cognitive + social
                
                # Update position
                particles[i] = particles[i] + velocities[i]
                
                # Normalize to unit hypersphere
                particles[i] = particles[i] / (np.linalg.norm(particles[i]) + 1e-10)
                
                # Evaluate fitness
                fitness = self._fitness(particles[i], class_label, X_encoded, y)
                
                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best_fitness[i] = fitness
                    personal_best_positions[i] = particles[i].copy()
                    
                    # Update global best
                    if fitness > global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = particles[i].copy()
        
        return global_best_position
    
    def fit(self, X, y):
        """
        Fit the hyperdimensional PSO classifier.
        
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
        
        # Initialize random projection for encoding
        rng = np.random.RandomState(self.random_state)
        self.projection_matrix_ = rng.randn(self.n_dimensions, self.n_features_in_)
        self.bias_ = rng.randn(self.n_dimensions)
        
        # Encode all training samples
        X_encoded = np.array([self._encode_sample(x) for x in X])
        
        # Optimize hypervector for each class using PSO
        self.class_hypervectors_ = {}
        for class_label in self.classes_:
            hypervector = self._optimize_class_hypervector(class_label, X_encoded, y)
            self.class_hypervectors_[class_label] = hypervector
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fitted
        check_is_fitted(self, ['class_hypervectors_', 'projection_matrix_'])
        
        # Validate input
        X = check_array(X)
        
        # Encode test samples
        X_encoded = np.array([self._encode_sample(x) for x in X])
        
        # Predict by finding most similar class hypervector
        predictions = []
        for sample in X_encoded:
            similarities = {}
            for class_label, class_hypervector in self.class_hypervectors_.items():
                similarities[class_label] = self._cosine_similarity(sample, class_hypervector)
            
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
            Test samples.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities (softmax of similarities).
        """
        check_is_fitted(self, ['class_hypervectors_', 'projection_matrix_'])
        X = check_array(X)
        
        X_encoded = np.array([self._encode_sample(x) for x in X])
        
        probabilities = []
        for sample in X_encoded:
            similarities = []
            for class_label in self.classes_:
                class_hypervector = self.class_hypervectors_[class_label]
                sim = self._cosine_similarity(sample, class_hypervector)
                similarities.append(sim)
            
            # Convert similarities to probabilities using softmax
            similarities = np.array(similarities)
            exp_sim = np.exp(similarities - np.max(similarities))
            proba = exp_sim / np.sum(exp_sim)
            probabilities.append(proba)
        
        return np.array(probabilities)