import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize


class DifferentiableTopologyClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier using continuous relaxation of Betti numbers via differentiable topology.
    
    This classifier learns optimal filtration functions through gradient-based optimization
    of smoothed topological features (relaxed Betti numbers) for classification tasks.
    
    Parameters
    ----------
    n_filtration_steps : int, default=20
        Number of filtration steps for persistent homology computation
    max_dimension : int, default=1
        Maximum homological dimension to compute (0 for components, 1 for loops)
    temperature : float, default=0.1
        Temperature parameter for sigmoid smoothing (smaller = sharper transitions)
    learning_rate : float, default=0.01
        Learning rate for gradient descent optimization
    n_iterations : int, default=100
        Number of optimization iterations
    regularization : float, default=0.01
        L2 regularization strength for filtration parameters
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_filtration_steps=20, max_dimension=1, temperature=0.1,
                 learning_rate=0.01, n_iterations=100, regularization=0.01,
                 random_state=None):
        self.n_filtration_steps = n_filtration_steps
        self.max_dimension = max_dimension
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.random_state = random_state
    
    def _smooth_sigmoid(self, x, threshold, temperature):
        """Smooth approximation of step function using sigmoid."""
        return 1.0 / (1.0 + np.exp(-(x - threshold) / temperature))
    
    def _compute_distance_matrix(self, X):
        """Compute pairwise distance matrix."""
        if len(X) == 1:
            return np.array([[0.0]])
        return squareform(pdist(X, metric='euclidean'))
    
    def _compute_relaxed_betti_numbers(self, distance_matrix, filtration_values):
        """
        Compute continuous relaxation of Betti numbers using differentiable topology.
        
        Uses smooth approximations of simplicial complex construction and boundary operators.
        """
        n_points = len(distance_matrix)
        n_steps = len(filtration_values)
        
        # Initialize Betti number sequences
        betti_sequences = np.zeros((self.max_dimension + 1, n_steps))
        
        for step_idx, threshold in enumerate(filtration_values):
            # Smooth edge weights based on distance threshold
            edge_weights = self._smooth_sigmoid(
                -distance_matrix, -threshold, self.temperature
            )
            
            # Betti_0: Connected components (relaxed)
            # Approximate using smooth graph Laplacian eigenvalues
            adjacency = edge_weights.copy()
            np.fill_diagonal(adjacency, 0)
            degree = np.sum(adjacency, axis=1)
            
            # Avoid division by zero
            degree_inv_sqrt = np.zeros_like(degree)
            nonzero_mask = degree > 1e-10
            degree_inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(degree[nonzero_mask])
            
            # Normalized Laplacian
            D_inv_sqrt = np.diag(degree_inv_sqrt)
            L_norm = np.eye(n_points) - D_inv_sqrt @ adjacency @ D_inv_sqrt
            
            # Eigenvalues approximate number of components
            eigenvalues = np.linalg.eigvalsh(L_norm)
            # Count near-zero eigenvalues (smoothly)
            betti_0 = np.sum(self._smooth_sigmoid(-eigenvalues, -0.1, self.temperature))
            betti_sequences[0, step_idx] = betti_0
            
            # Betti_1: Loops (relaxed)
            if self.max_dimension >= 1:
                # Approximate using cycle rank formula: edges - vertices + components
                n_edges_smooth = np.sum(edge_weights) / 2  # Undirected graph
                betti_1 = np.maximum(0, n_edges_smooth - n_points + betti_0)
                betti_sequences[1, step_idx] = betti_1
        
        return betti_sequences.flatten()
    
    def _point_cloud_from_features(self, feature_vector, n_points=10):
        """
        Convert a feature vector to a point cloud for topological analysis.
        
        Uses a sliding window approach to create points in 2D space.
        """
        n_features = len(feature_vector)
        
        if n_features < 2:
            # Handle edge case: create minimal point cloud
            points = np.array([[0, 0], [1, 0]])
        else:
            # Create points using consecutive feature pairs
            n_points = min(n_points, n_features)
            points = []
            
            for i in range(n_points):
                idx1 = (i * n_features) // n_points
                idx2 = ((i + 1) * n_features) // n_points
                if idx2 >= n_features:
                    idx2 = n_features - 1
                
                x = feature_vector[idx1]
                y = feature_vector[idx2] if idx1 != idx2 else feature_vector[idx1] * 0.1
                points.append([x, y])
            
            points = np.array(points)
            
            # Add small noise to avoid degenerate configurations
            points += np.random.RandomState(self.random_state).randn(*points.shape) * 1e-6
        
        return points
    
    def _extract_topological_features(self, X, filtration_params):
        """Extract topological features for a dataset."""
        features_list = []
        
        for sample in X:
            # Convert feature vector to point cloud
            point_cloud = self._point_cloud_from_features(sample)
            
            distance_matrix = self._compute_distance_matrix(point_cloud)
            
            # Create filtration values
            max_dist = np.max(distance_matrix) if np.max(distance_matrix) > 0 else 1.0
            base_filtration = np.linspace(0, max_dist * 1.5, self.n_filtration_steps)
            
            # Apply learned filtration scaling
            filtration_values = base_filtration * np.exp(filtration_params[0])
            
            betti_features = self._compute_relaxed_betti_numbers(
                distance_matrix, filtration_values
            )
            features_list.append(betti_features)
        
        return np.array(features_list)
    
    def _compute_loss_and_gradient(self, params, X, y, n_classes):
        """Compute loss for optimization."""
        n_features = (self.max_dimension + 1) * self.n_filtration_steps
        
        # Split parameters
        filtration_params = params[:1]
        classifier_weights = params[1:].reshape(n_classes, n_features)
        
        # Extract features
        features = self._extract_topological_features(X, filtration_params)
        
        # Compute logits
        logits = features @ classifier_weights.T
        
        # Softmax with numerical stability
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Cross-entropy loss
        n_samples = len(y)
        log_probs = np.log(probs[np.arange(n_samples), y] + 1e-10)
        loss = -np.mean(log_probs)
        
        # Add regularization
        loss += self.regularization * np.sum(params ** 2)
        
        return loss
    
    def fit(self, X, y):
        """
        Fit the differentiable topology classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels
        
        Returns
        -------
        self : object
            Fitted classifier
        """
        X, y = check_X_y(X, y, accept_sparse=False)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        
        # Map labels to indices
        self.label_map_ = {label: idx for idx, label in enumerate(self.classes_)}
        y_mapped = np.array([self.label_map_[label] for label in y])
        
        # Initialize parameters
        rng = np.random.RandomState(self.random_state)
        n_features = (self.max_dimension + 1) * self.n_filtration_steps
        
        # Initialize: filtration_param and classifier_weights
        filtration_param = np.array([0.0])
        classifier_weights = rng.randn(self.n_classes_, n_features) * 0.1
        
        params = np.concatenate([
            filtration_param,
            classifier_weights.flatten()
        ])
        
        # Optimize using scipy with reduced iterations for stability
        try:
            result = minimize(
                fun=lambda p: self._compute_loss_and_gradient(p, X, y_mapped, self.n_classes_),
                x0=params,
                method='L-BFGS-B',
                options={'maxiter': self.n_iterations, 'disp': False, 'ftol': 1e-6}
            )
            
            # Store learned parameters
            self.filtration_param_ = result.x[:1]
            self.classifier_weights_ = result.x[1:].reshape(
                self.n_classes_, n_features
            )
        except Exception as e:
            # Fallback: use initial parameters if optimization fails
            print(f"Optimization failed: {e}. Using initial parameters.")
            self.filtration_param_ = filtration_param
            self.classifier_weights_ = classifier_weights
        
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=False)
        
        # Extract features
        features = self._extract_topological_features(X, self.filtration_param_)
        
        # Compute logits and probabilities
        logits = features @ self.classifier_weights_.T
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        return probs
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return self.classes_[class_indices]