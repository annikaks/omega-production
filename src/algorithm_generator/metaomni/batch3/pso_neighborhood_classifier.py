import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import cdist
from scipy.stats import entropy


class PSONeighborhoodClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses Particle Swarm Optimization to evolve neighborhood
    function parameters that minimize classification entropy across a map.
    
    Parameters
    ----------
    map_size : tuple, default=(10, 10)
        Size of the self-organizing map (rows, cols).
    n_particles : int, default=20
        Number of particles in the PSO swarm.
    n_iterations : int, default=50
        Number of PSO iterations.
    w : float, default=0.7
        Inertia weight for PSO.
    c1 : float, default=1.5
        Cognitive parameter for PSO.
    c2 : float, default=1.5
        Social parameter for PSO.
    neighborhood_type : str, default='gaussian'
        Type of neighborhood function ('gaussian' or 'bubble').
    random_state : int, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    map_ : ndarray of shape (map_size[0], map_size[1], n_features)
        The trained self-organizing map.
    map_labels_ : ndarray of shape (map_size[0], map_size[1])
        Class labels assigned to each map node.
    best_params_ : dict
        Best neighborhood parameters found by PSO.
    classes_ : ndarray
        Unique class labels.
    """
    
    def __init__(self, map_size=(10, 10), n_particles=20, n_iterations=50,
                 w=0.7, c1=1.5, c2=1.5, neighborhood_type='gaussian',
                 random_state=None):
        self.map_size = map_size
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.neighborhood_type = neighborhood_type
        self.random_state = random_state
    
    def _initialize_map(self, X):
        """Initialize the SOM with random weights from data range."""
        rng = np.random.RandomState(self.random_state)
        n_features = X.shape[1]
        map_shape = (self.map_size[0], self.map_size[1], n_features)
        
        # Initialize with random values in the range of the data
        self.map_ = rng.uniform(
            X.min(axis=0),
            X.max(axis=0),
            size=map_shape
        )
    
    def _get_bmu(self, x):
        """Find the Best Matching Unit for a given input vector."""
        map_2d = self.map_.reshape(-1, self.map_.shape[2])
        distances = np.linalg.norm(map_2d - x, axis=1)
        bmu_idx = np.argmin(distances)
        bmu_row = bmu_idx // self.map_size[1]
        bmu_col = bmu_idx % self.map_size[1]
        return bmu_row, bmu_col
    
    def _neighborhood_function(self, bmu_pos, params, iteration, max_iter):
        """Calculate neighborhood influence based on parameters."""
        sigma_start = params['sigma_start']
        sigma_end = params['sigma_end']
        learning_rate_start = params['lr_start']
        learning_rate_end = params['lr_end']
        
        # Decay sigma and learning rate
        sigma = sigma_start * (sigma_end / sigma_start) ** (iteration / max_iter)
        lr = learning_rate_start * (learning_rate_end / learning_rate_start) ** (iteration / max_iter)
        
        # Create grid of positions
        rows, cols = np.meshgrid(
            np.arange(self.map_size[0]),
            np.arange(self.map_size[1]),
            indexing='ij'
        )
        
        # Calculate distances from BMU
        distances = np.sqrt(
            (rows - bmu_pos[0])**2 + (cols - bmu_pos[1])**2
        )
        
        if self.neighborhood_type == 'gaussian':
            influence = np.exp(-(distances**2) / (2 * sigma**2))
        else:  # bubble
            influence = (distances <= sigma).astype(float)
        
        return influence * lr
    
    def _train_som(self, X, params, n_epochs=100):
        """Train the SOM with given neighborhood parameters."""
        rng = np.random.RandomState(self.random_state)
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = rng.permutation(len(X))
            
            for idx in indices:
                x = X[idx]
                
                # Find BMU
                bmu_row, bmu_col = self._get_bmu(x)
                
                # Calculate neighborhood influence
                influence = self._neighborhood_function(
                    (bmu_row, bmu_col), params, epoch, n_epochs
                )
                
                # Update weights
                for i in range(self.map_size[0]):
                    for j in range(self.map_size[1]):
                        self.map_[i, j] += influence[i, j] * (x - self.map_[i, j])
    
    def _assign_labels_to_map(self, X, y):
        """Assign class labels to each map node based on training data."""
        self.map_labels_ = np.zeros(self.map_size, dtype=int)
        label_counts = [[{} for _ in range(self.map_size[1])] 
                       for _ in range(self.map_size[0])]
        
        # Count class occurrences for each node
        for x, label in zip(X, y):
            bmu_row, bmu_col = self._get_bmu(x)
            if label not in label_counts[bmu_row][bmu_col]:
                label_counts[bmu_row][bmu_col][label] = 0
            label_counts[bmu_row][bmu_col][label] += 1
        
        # Assign most common label to each node
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if label_counts[i][j]:
                    self.map_labels_[i, j] = max(
                        label_counts[i][j],
                        key=label_counts[i][j].get
                    )
    
    def _calculate_classification_entropy(self, X, y):
        """Calculate average classification entropy across the map."""
        node_samples = [[[] for _ in range(self.map_size[1])] 
                       for _ in range(self.map_size[0])]
        
        # Collect samples for each node
        for x, label in zip(X, y):
            bmu_row, bmu_col = self._get_bmu(x)
            node_samples[bmu_row][bmu_col].append(label)
        
        # Calculate entropy for each node
        total_entropy = 0
        node_count = 0
        
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if node_samples[i][j]:
                    labels = node_samples[i][j]
                    unique, counts = np.unique(labels, return_counts=True)
                    probs = counts / len(labels)
                    node_entropy = entropy(probs)
                    total_entropy += node_entropy
                    node_count += 1
        
        return total_entropy / max(node_count, 1)
    
    def _pso_optimize(self, X, y):
        """Use PSO to find optimal neighborhood parameters."""
        rng = np.random.RandomState(self.random_state)
        
        # Parameter bounds: [sigma_start, sigma_end, lr_start, lr_end]
        bounds_min = np.array([0.5, 0.1, 0.01, 0.001])
        bounds_max = np.array([5.0, 1.0, 1.0, 0.1])
        
        # Initialize particles
        particles = rng.uniform(
            bounds_min, bounds_max, (self.n_particles, 4)
        )
        velocities = rng.uniform(-0.1, 0.1, (self.n_particles, 4))
        
        # Initialize personal and global bests
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.n_particles, np.inf)
        global_best_position = None
        global_best_score = np.inf
        
        # PSO iterations
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # Create parameter dict
                params = {
                    'sigma_start': particles[i, 0],
                    'sigma_end': particles[i, 1],
                    'lr_start': particles[i, 2],
                    'lr_end': particles[i, 3]
                }
                
                # Ensure sigma_start > sigma_end and lr_start > lr_end
                if params['sigma_start'] <= params['sigma_end']:
                    params['sigma_start'] = params['sigma_end'] + 0.1
                if params['lr_start'] <= params['lr_end']:
                    params['lr_start'] = params['lr_end'] + 0.01
                
                # Train SOM with these parameters
                self._initialize_map(X)
                self._train_som(X, params, n_epochs=50)
                
                # Calculate fitness (classification entropy)
                score = self._calculate_classification_entropy(X, y)
                
                # Update personal best
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = particles[i].copy()
                
                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particles[i].copy()
            
            # Update velocities and positions
            for i in range(self.n_particles):
                r1, r2 = rng.random(2)
                
                velocities[i] = (
                    self.w * velocities[i] +
                    self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                    self.c2 * r2 * (global_best_position - particles[i])
                )
                
                particles[i] += velocities[i]
                
                # Enforce bounds
                particles[i] = np.clip(particles[i], bounds_min, bounds_max)
        
        # Return best parameters
        return {
            'sigma_start': global_best_position[0],
            'sigma_end': global_best_position[1],
            'lr_start': global_best_position[2],
            'lr_end': global_best_position[3]
        }
    
    def fit(self, X_train, y_train):
        """
        Fit the classifier using PSO to optimize neighborhood parameters.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data.
        y_train : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        self.classes_ = unique_labels(y_train)
        
        # Use PSO to find optimal parameters
        self.best_params_ = self._pso_optimize(X_train, y_train)
        
        # Train final SOM with best parameters
        self._initialize_map(X_train)
        self._train_som(X_train, self.best_params_, n_epochs=100)
        
        # Assign labels to map nodes
        self._assign_labels_to_map(X_train, y_train)
        
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
        # Check if fitted
        check_is_fitted(self, ['map_', 'map_labels_', 'best_params_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Predict labels
        predictions = []
        for x in X_test:
            bmu_row, bmu_col = self._get_bmu(x)
            predictions.append(self.map_labels_[bmu_row, bmu_col])
        
        return np.array(predictions)