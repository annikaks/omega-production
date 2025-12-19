import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import cdist


class FisherAntColonyClassifier(BaseEstimator, ClassifierMixin):
    """
    Bio-inspired Ant Colony Optimization classifier that traverses Fisher information
    manifold using pheromone trails to mark maximally informative geodesic routes
    between class distributions.
    
    Parameters
    ----------
    n_ants : int, default=50
        Number of ants in the colony
    n_iterations : int, default=100
        Number of optimization iterations
    alpha : float, default=1.0
        Pheromone importance factor
    beta : float, default=2.0
        Heuristic information importance factor
    evaporation_rate : float, default=0.1
        Pheromone evaporation rate (0 to 1)
    pheromone_deposit : float, default=1.0
        Amount of pheromone deposited by ants
    n_landmarks : int, default=10
        Number of landmark points per class for manifold discretization
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_ants=50, n_iterations=100, alpha=1.0, beta=2.0,
                 evaporation_rate=0.1, pheromone_deposit=1.0, n_landmarks=10,
                 random_state=None):
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.n_landmarks = n_landmarks
        self.random_state = random_state
    
    def _compute_fisher_information(self, X, y, class_label):
        """Compute Fisher Information Matrix for a class distribution."""
        X_class = X[y == class_label]
        if len(X_class) < 2:
            return np.eye(X.shape[1]) * 1e-3
        
        # Estimate covariance (Fisher information is inverse of covariance for Gaussian)
        cov = np.cov(X_class.T, bias=True)
        
        # Add regularization for numerical stability
        reg = np.eye(X.shape[1]) * 1e-4
        cov = cov + reg
        
        # Compute Fisher information matrix
        try:
            fisher_info = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # If inversion fails, use pseudo-inverse
            fisher_info = np.linalg.pinv(cov)
        
        return fisher_info
    
    def _fisher_distance(self, point1, point2, fisher_info):
        """Compute Fisher-Rao distance between two points on the manifold."""
        diff = point1 - point2
        
        # Ensure diff is 1D
        if diff.ndim > 1:
            diff = diff.flatten()
        
        # Compute quadratic form
        distance = np.sqrt(np.abs(diff @ fisher_info @ diff))
        return distance
    
    def _create_landmarks(self, X, y):
        """Create landmark points for each class to discretize the manifold."""
        landmarks = {}
        landmark_classes = {}
        
        for class_label in self.classes_:
            X_class = X[y == class_label]
            n_samples = len(X_class)
            
            if n_samples <= self.n_landmarks:
                landmarks[class_label] = X_class.copy()
            else:
                # Use stratified sampling to select representative landmarks
                indices = self.rng_.choice(n_samples, self.n_landmarks, replace=False)
                landmarks[class_label] = X_class[indices]
            
            landmark_classes[class_label] = np.full(len(landmarks[class_label]), class_label)
        
        return landmarks, landmark_classes
    
    def _initialize_pheromones(self, n_nodes):
        """Initialize pheromone matrix."""
        return np.ones((n_nodes, n_nodes)) * 0.1
    
    def _compute_heuristic_info(self, landmarks_array, fisher_infos):
        """Compute heuristic information based on Fisher distances."""
        n_nodes = len(landmarks_array)
        heuristic = np.zeros((n_nodes, n_nodes))
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    # Use average Fisher information for distance computation
                    avg_fisher = (fisher_infos[self.landmark_labels_[i]] + 
                                 fisher_infos[self.landmark_labels_[j]]) / 2
                    dist = self._fisher_distance(landmarks_array[i], 
                                                 landmarks_array[j], 
                                                 avg_fisher)
                    heuristic[i, j] = 1.0 / (dist + 1e-6)
                else:
                    heuristic[i, j] = 0.0
        
        return heuristic
    
    def _ant_solution(self, start_node, target_class, pheromones, heuristic):
        """Construct a solution path for a single ant."""
        current_node = start_node
        path = [current_node]
        visited = {current_node}
        
        max_steps = min(20, len(self.landmarks_array_))
        
        for _ in range(max_steps):
            # Find nodes of target class
            target_nodes = np.where(self.landmark_labels_ == target_class)[0]
            
            if current_node in target_nodes:
                break
            
            # Calculate transition probabilities
            unvisited = [n for n in range(len(self.landmarks_array_)) if n not in visited]
            
            if not unvisited:
                break
            
            probabilities = []
            for next_node in unvisited:
                pheromone = pheromones[current_node, next_node] ** self.alpha
                heuristic_val = heuristic[current_node, next_node] ** self.beta
                
                # Bias towards target class
                if self.landmark_labels_[next_node] == target_class:
                    heuristic_val *= 2.0
                
                probabilities.append(pheromone * heuristic_val)
            
            probabilities = np.array(probabilities)
            prob_sum = probabilities.sum()
            
            # Handle edge case where all probabilities are zero
            if prob_sum < 1e-10:
                probabilities = np.ones(len(unvisited)) / len(unvisited)
            else:
                probabilities = probabilities / prob_sum
            
            # Ensure probabilities sum to exactly 1.0 (fix floating point errors)
            probabilities = probabilities / probabilities.sum()
            
            # Select next node
            next_node = self.rng_.choice(unvisited, p=probabilities)
            path.append(next_node)
            visited.add(next_node)
            current_node = next_node
        
        return path
    
    def _evaluate_path(self, path, target_class):
        """Evaluate path quality based on reaching target and Fisher information."""
        if len(path) < 2:
            return 0.0
        
        # Reward reaching target class
        final_class = self.landmark_labels_[path[-1]]
        class_reward = 10.0 if final_class == target_class else 0.0
        
        # Penalize path length
        length_penalty = len(path) * 0.1
        
        # Reward shorter paths to target
        if final_class == target_class:
            efficiency_bonus = 5.0 / len(path)
        else:
            efficiency_bonus = 0.0
        
        return class_reward + efficiency_bonus - length_penalty
    
    def fit(self, X, y):
        """
        Fit the Fisher Ant Colony classifier.
        
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
        
        # Initialize random state
        self.rng_ = np.random.RandomState(self.random_state)
        
        # Compute Fisher information matrices for each class
        self.fisher_infos_ = {}
        for class_label in self.classes_:
            self.fisher_infos_[class_label] = self._compute_fisher_information(X, y, class_label)
        
        # Create landmarks for manifold discretization
        landmarks, landmark_classes = self._create_landmarks(X, y)
        
        # Flatten landmarks into array
        self.landmarks_array_ = np.vstack([landmarks[c] for c in self.classes_])
        self.landmark_labels_ = np.hstack([landmark_classes[c] for c in self.classes_])
        
        n_nodes = len(self.landmarks_array_)
        
        # Initialize pheromones and heuristic information
        pheromones = self._initialize_pheromones(n_nodes)
        heuristic = self._compute_heuristic_info(self.landmarks_array_, self.fisher_infos_)
        
        # Ant Colony Optimization
        best_paths = {c: [] for c in self.classes_}
        
        for iteration in range(self.n_iterations):
            all_paths = []
            
            # Each ant constructs solutions for each class
            for _ in range(self.n_ants):
                for target_class in self.classes_:
                    # Start from random node
                    start_node = self.rng_.randint(0, n_nodes)
                    path = self._ant_solution(start_node, target_class, pheromones, heuristic)
                    quality = self._evaluate_path(path, target_class)
                    all_paths.append((path, quality, target_class))
            
            # Update best paths
            for path, quality, target_class in all_paths:
                if quality > 0:
                    best_paths[target_class].append((path, quality))
            
            # Keep only top paths per class
            for target_class in self.classes_:
                if len(best_paths[target_class]) > 100:
                    best_paths[target_class] = sorted(
                        best_paths[target_class], 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:100]
            
            # Evaporate pheromones
            pheromones *= (1 - self.evaporation_rate)
            
            # Deposit pheromones
            for path, quality, _ in all_paths:
                if quality > 0:
                    deposit = self.pheromone_deposit * quality
                    for i in range(len(path) - 1):
                        pheromones[path[i], path[i+1]] += deposit
                        pheromones[path[i+1], path[i]] += deposit
        
        self.pheromones_ = pheromones
        self.best_paths_ = best_paths
        
        # Store class centroids and statistics for prediction
        self.class_centroids_ = {}
        self.class_covs_ = {}
        
        for class_label in self.classes_:
            X_class = X[y == class_label]
            self.class_centroids_[class_label] = np.mean(X_class, axis=0)
            
            if len(X_class) > 1:
                self.class_covs_[class_label] = np.cov(X_class.T, bias=True) + np.eye(X.shape[1]) * 1e-4
            else:
                self.class_covs_[class_label] = np.eye(X.shape[1]) * 1e-3
        
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
        check_is_fitted(self)
        X = check_array(X)
        
        predictions = []
        
        for x in X:
            # Find nearest landmarks
            distances = cdist([x], self.landmarks_array_)[0]
            k_nearest = min(5, len(distances))
            nearest_indices = np.argsort(distances)[:k_nearest]
            
            # Compute class scores based on pheromone trails and Fisher distance
            class_scores = {}
            
            for class_label in self.classes_:
                # Find landmarks of this class
                class_landmarks = np.where(self.landmark_labels_ == class_label)[0]
                
                if len(class_landmarks) == 0:
                    class_scores[class_label] = 0.0
                    continue
                
                # Score based on pheromone strength from nearest landmarks to class landmarks
                pheromone_score = 0.0
                for nearest_idx in nearest_indices:
                    for cl in class_landmarks:
                        pheromone_score += self.pheromones_[nearest_idx, cl]
                
                pheromone_score /= (len(nearest_indices) * len(class_landmarks))
                
                # Score based on Fisher distance to class centroid
                fisher_info = self.fisher_infos_[class_label]
                fisher_dist = self._fisher_distance(x, self.class_centroids_[class_label], 
                                                   fisher_info)
                distance_score = 1.0 / (fisher_dist + 1e-6)
                
                # Score based on Mahalanobis distance
                try:
                    diff = x - self.class_centroids_[class_label]
                    cov_inv = np.linalg.inv(self.class_covs_[class_label])
                    mahal_dist = np.sqrt(diff @ cov_inv @ diff)
                    mahal_score = 1.0 / (mahal_dist + 1e-6)
                except:
                    mahal_score = 0.0
                
                # Combined score with weights
                class_scores[class_label] = (
                    0.4 * pheromone_score + 
                    0.4 * distance_score + 
                    0.2 * mahal_score
                )
            
            # Predict class with highest score
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
        
        return np.array(predictions)