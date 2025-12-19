import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
import warnings


class AntColonyHomologyClassifier(BaseEstimator, ClassifierMixin):
    """
    Bio-inspired Ant Colony Optimization classifier for topological data analysis.
    
    Uses swarm intelligence to discover optimal homology class weightings by
    traversing the filtration parameter space. Ants explore different combinations
    of persistence features and their weights to find optimal classification boundaries.
    
    Parameters
    ----------
    n_ants : int, default=20
        Number of ants in the colony
    n_iterations : int, default=50
        Number of optimization iterations
    alpha : float, default=1.0
        Pheromone importance factor
    beta : float, default=2.0
        Heuristic information importance factor
    evaporation_rate : float, default=0.1
        Pheromone evaporation rate (0 to 1)
    q : float, default=1.0
        Pheromone deposit factor
    max_filtration : float, default=2.0
        Maximum filtration value for persistence computation
    n_filtration_steps : int, default=10
        Number of filtration steps to consider
    homology_dims : list, default=[0, 1]
        Homology dimensions to compute
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_ants=20, n_iterations=50, alpha=1.0, beta=2.0,
                 evaporation_rate=0.1, q=1.0, max_filtration=2.0,
                 n_filtration_steps=10, homology_dims=None, random_state=None):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.max_filtration = max_filtration
        self.n_filtration_steps = n_filtration_steps
        self.homology_dims = homology_dims if homology_dims is not None else [0, 1]
        self.random_state = random_state
        
    def _compute_persistence_features(self, X):
        """Compute simplified persistence features from point cloud data."""
        n_samples = X.shape[0]
        features_list = []
        
        for i in range(n_samples):
            point_cloud = X[i].reshape(-1, X.shape[1]) if X[i].ndim == 1 else X[i]
            
            # Compute pairwise distances
            if point_cloud.shape[0] > 1:
                distances = pdist(point_cloud.reshape(-1, 1) if point_cloud.ndim == 1 
                                 else point_cloud)
                dist_matrix = squareform(distances)
            else:
                dist_matrix = np.array([[0]])
            
            # Simplified persistence computation across filtration values
            filtration_values = np.linspace(0, self.max_filtration, self.n_filtration_steps)
            persistence_features = []
            
            for filt_val in filtration_values:
                # H0: Connected components (simplified)
                if 0 in self.homology_dims:
                    # Count components based on distance threshold
                    adj_matrix = dist_matrix < filt_val
                    n_components = self._count_components(adj_matrix)
                    persistence_features.append(n_components)
                
                # H1: Loops (simplified heuristic based on triangles)
                if 1 in self.homology_dims:
                    n_loops = self._estimate_loops(dist_matrix, filt_val)
                    persistence_features.append(n_loops)
            
            # Add statistical features
            if dist_matrix.size > 1:
                persistence_features.extend([
                    np.mean(dist_matrix[dist_matrix > 0]),
                    np.std(dist_matrix[dist_matrix > 0]),
                    np.max(dist_matrix),
                    np.percentile(dist_matrix[dist_matrix > 0], 75) if np.any(dist_matrix > 0) else 0
                ])
            else:
                persistence_features.extend([0, 0, 0, 0])
            
            features_list.append(persistence_features)
        
        return np.array(features_list)
    
    def _count_components(self, adj_matrix):
        """Count connected components using union-find."""
        n = adj_matrix.shape[0]
        if n == 0:
            return 0
        
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j]:
                    union(i, j)
        
        return len(set(find(i) for i in range(n)))
    
    def _estimate_loops(self, dist_matrix, threshold):
        """Estimate number of loops based on triangles in the graph."""
        n = dist_matrix.shape[0]
        if n < 3:
            return 0
        
        loop_count = 0
        for i, j, k in combinations(range(n), 3):
            if (dist_matrix[i, j] < threshold and 
                dist_matrix[j, k] < threshold and 
                dist_matrix[i, k] < threshold):
                loop_count += 1
        
        return min(loop_count, n)  # Normalize
    
    def _initialize_pheromones(self, n_features):
        """Initialize pheromone matrix."""
        return np.ones(n_features) * 0.1
    
    def _compute_heuristic(self, features, labels):
        """Compute heuristic information based on feature-class correlation."""
        heuristic = np.zeros(features.shape[1])
        
        for i in range(features.shape[1]):
            # Compute correlation between feature and labels
            if np.std(features[:, i]) > 1e-10:
                correlation = np.abs(np.corrcoef(features[:, i], labels)[0, 1])
                heuristic[i] = correlation if not np.isnan(correlation) else 0.01
            else:
                heuristic[i] = 0.01
        
        return heuristic + 1e-10  # Avoid zeros
    
    def _ant_solution(self, pheromones, heuristic, rng):
        """Generate ant solution (feature weights)."""
        n_features = len(pheromones)
        
        # Compute probabilities
        probabilities = (pheromones ** self.alpha) * (heuristic ** self.beta)
        probabilities = probabilities / (np.sum(probabilities) + 1e-10)
        
        # Generate weights based on probabilities
        weights = probabilities + rng.uniform(0, 0.1, n_features)
        weights = weights / (np.sum(weights) + 1e-10)
        
        return weights
    
    def _evaluate_solution(self, weights, features, labels):
        """Evaluate solution quality using weighted features."""
        weighted_features = features * weights
        
        # Compute class centroids
        classes = np.unique(labels)
        centroids = {}
        for cls in classes:
            mask = labels == cls
            centroids[cls] = np.mean(weighted_features[mask], axis=0)
        
        # Compute classification accuracy (simplified)
        correct = 0
        for i, sample in enumerate(weighted_features):
            distances = {cls: np.linalg.norm(sample - centroid) 
                        for cls, centroid in centroids.items()}
            predicted = min(distances, key=distances.get)
            if predicted == labels[i]:
                correct += 1
        
        return correct / len(labels)
    
    def fit(self, X, y):
        """
        Fit the ant colony optimization classifier.
        
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
        X, y = check_X_y(X, y, accept_sparse=False)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        rng = np.random.RandomState(self.random_state)
        
        # Compute persistence features
        self.persistence_features_ = self._compute_persistence_features(X)
        n_features = self.persistence_features_.shape[1]
        
        # Initialize ACO components
        pheromones = self._initialize_pheromones(n_features)
        heuristic = self._compute_heuristic(self.persistence_features_, y)
        
        best_weights = None
        best_fitness = 0
        
        # ACO main loop
        for iteration in range(self.n_iterations):
            iteration_solutions = []
            iteration_fitness = []
            
            # Each ant constructs a solution
            for ant in range(self.n_ants):
                weights = self._ant_solution(pheromones, heuristic, rng)
                fitness = self._evaluate_solution(weights, self.persistence_features_, y)
                
                iteration_solutions.append(weights)
                iteration_fitness.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_weights = weights.copy()
            
            # Evaporate pheromones
            pheromones *= (1 - self.evaporation_rate)
            
            # Deposit pheromones
            for weights, fitness in zip(iteration_solutions, iteration_fitness):
                pheromones += self.q * fitness * weights
            
            # Normalize pheromones
            pheromones = np.clip(pheromones, 0.01, 10.0)
        
        # Store best solution
        self.best_weights_ = best_weights if best_weights is not None else np.ones(n_features) / n_features
        
        # Compute class centroids with best weights
        weighted_features = self.persistence_features_ * self.best_weights_
        self.centroids_ = {}
        for cls in self.classes_:
            mask = y == cls
            self.centroids_[cls] = np.mean(weighted_features[mask], axis=0)
        
        self.is_fitted_ = True
        return self
    
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
        check_is_fitted(self, ['is_fitted_', 'best_weights_', 'centroids_'])
        X = check_array(X, accept_sparse=False)
        
        # Compute persistence features for test data
        test_features = self._compute_persistence_features(X)
        weighted_features = test_features * self.best_weights_
        
        # Predict based on nearest centroid
        predictions = []
        for sample in weighted_features:
            distances = {cls: np.linalg.norm(sample - centroid)
                        for cls, centroid in self.centroids_.items()}
            predicted = min(distances, key=distances.get)
            predictions.append(predicted)
        
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
        check_is_fitted(self, ['is_fitted_', 'best_weights_', 'centroids_'])
        X = check_array(X, accept_sparse=False)
        
        test_features = self._compute_persistence_features(X)
        weighted_features = test_features * self.best_weights_
        
        probas = []
        for sample in weighted_features:
            distances = np.array([np.linalg.norm(sample - centroid)
                                 for centroid in self.centroids_.values()])
            # Convert distances to probabilities (inverse distance weighting)
            inv_distances = 1.0 / (distances + 1e-10)
            proba = inv_distances / np.sum(inv_distances)
            probas.append(proba)
        
        return np.array(probas)