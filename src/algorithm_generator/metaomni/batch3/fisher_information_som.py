import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal


class FisherInformationSOM(BaseEstimator, ClassifierMixin):
    """
    Self-Organizing Map with Fisher Information-based lattice warping.
    
    Uses Fisher information metric to warp the SOM lattice geometry,
    allocating more neurons to regions with steeper information gradients.
    
    Parameters
    ----------
    n_neurons : int, default=100
        Number of neurons in the SOM lattice
    n_iterations : int, default=1000
        Number of training iterations
    learning_rate : float, default=0.5
        Initial learning rate
    sigma : float, default=1.0
        Initial neighborhood radius
    fisher_weight : float, default=0.5
        Weight for Fisher information in lattice warping (0-1)
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_neurons=100, n_iterations=1000, learning_rate=0.5,
                 sigma=1.0, fisher_weight=0.5, random_state=None):
        self.n_neurons = n_neurons
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.fisher_weight = fisher_weight
        self.random_state = random_state
    
    def _compute_fisher_information(self, X, y):
        """
        Compute Fisher information matrix for each data point.
        
        Fisher information measures the amount of information that an observable
        random variable carries about an unknown parameter.
        """
        n_samples, n_features = X.shape
        fisher_scores = np.zeros(n_samples)
        
        # Compute class-conditional statistics
        for class_label in self.classes_:
            mask = y == class_label
            X_class = X[mask]
            
            if len(X_class) < 2:
                continue
            
            # Estimate mean and covariance for this class
            mean = np.mean(X_class, axis=0)
            cov = np.cov(X_class.T) + 1e-6 * np.eye(n_features)
            
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov)
            
            # Compute Fisher information score for each point
            for i in range(n_samples):
                diff = X[i] - mean
                # Fisher information approximation using Mahalanobis distance
                fisher_scores[i] += np.dot(diff, np.dot(cov_inv, diff))
        
        # Normalize scores
        fisher_scores = (fisher_scores - fisher_scores.min()) / (fisher_scores.max() - fisher_scores.min() + 1e-10)
        
        return fisher_scores
    
    def _initialize_lattice(self, X, fisher_scores):
        """
        Initialize SOM lattice with Fisher information-based warping.
        """
        n_samples, n_features = X.shape
        
        # Create base lattice positions in 2D
        grid_size = int(np.ceil(np.sqrt(self.n_neurons)))
        x_coords = np.linspace(0, 1, grid_size)
        y_coords = np.linspace(0, 1, grid_size)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        lattice_positions = np.column_stack([xx.ravel(), yy.ravel()])[:self.n_neurons]
        
        # Warp lattice based on Fisher information
        # Allocate more neurons to high Fisher information regions
        if self.fisher_weight > 0:
            # Create density map based on Fisher scores
            n_bins = 20
            hist, x_edges, y_edges = np.histogram2d(
                X[:, 0], X[:, 1] if n_features > 1 else X[:, 0],
                bins=n_bins, weights=fisher_scores
            )
            
            # Normalize histogram
            hist = hist / (hist.sum() + 1e-10)
            
            # Warp lattice positions towards high-density regions
            for i in range(self.n_neurons):
                # Find nearest high-information region
                x_idx = int(lattice_positions[i, 0] * (n_bins - 1))
                y_idx = int(lattice_positions[i, 1] * (n_bins - 1))
                x_idx = np.clip(x_idx, 0, n_bins - 1)
                y_idx = np.clip(y_idx, 0, n_bins - 1)
                
                # Apply warping based on local Fisher information
                warp_factor = hist[x_idx, y_idx] * self.fisher_weight
                lattice_positions[i] += np.random.randn(2) * warp_factor * 0.1
        
        self.lattice_positions_ = lattice_positions
        
        # Initialize neuron weights randomly from data distribution
        indices = self.rng_.choice(n_samples, size=self.n_neurons, replace=True,
                                   p=fisher_scores / fisher_scores.sum())
        self.weights_ = X[indices].copy()
        self.weights_ += self.rng_.randn(self.n_neurons, n_features) * 0.01
    
    def _find_bmu(self, x):
        """Find Best Matching Unit for input vector x."""
        distances = np.linalg.norm(self.weights_ - x, axis=1)
        return np.argmin(distances)
    
    def _update_weights(self, x, bmu_idx, iteration):
        """Update neuron weights based on BMU and neighborhood."""
        # Decay learning rate and neighborhood radius
        lr = self.learning_rate * np.exp(-iteration / self.n_iterations)
        sigma_t = self.sigma * np.exp(-iteration / self.n_iterations)
        
        # Compute neighborhood influence
        distances = np.linalg.norm(
            self.lattice_positions_ - self.lattice_positions_[bmu_idx],
            axis=1
        )
        influence = np.exp(-distances**2 / (2 * sigma_t**2))
        
        # Update weights
        for i in range(self.n_neurons):
            self.weights_[i] += lr * influence[i] * (x - self.weights_[i])
    
    def fit(self, X, y):
        """
        Fit the Fisher Information SOM classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels
        
        Returns
        -------
        self : object
            Fitted estimator
        """
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        self.rng_ = np.random.RandomState(self.random_state)
        
        # Compute Fisher information for each sample
        fisher_scores = self._compute_fisher_information(X, y)
        
        # Initialize lattice with Fisher information warping
        self._initialize_lattice(X, fisher_scores)
        
        # Train SOM with weighted sampling based on Fisher information
        sample_weights = fisher_scores / fisher_scores.sum()
        
        for iteration in range(self.n_iterations):
            # Sample with probability proportional to Fisher information
            idx = self.rng_.choice(len(X), p=sample_weights)
            x = X[idx]
            
            # Find BMU and update weights
            bmu_idx = self._find_bmu(x)
            self._update_weights(x, bmu_idx, iteration)
        
        # Assign class labels to neurons based on training data
        self.neuron_labels_ = np.zeros(self.n_neurons, dtype=int)
        self.neuron_class_counts_ = np.zeros((self.n_neurons, len(self.classes_)))
        
        for i in range(len(X)):
            bmu_idx = self._find_bmu(X[i])
            class_idx = np.where(self.classes_ == y[i])[0][0]
            self.neuron_class_counts_[bmu_idx, class_idx] += 1
        
        # Assign most frequent class to each neuron
        for i in range(self.n_neurons):
            if self.neuron_class_counts_[i].sum() > 0:
                self.neuron_labels_[i] = np.argmax(self.neuron_class_counts_[i])
        
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
        check_is_fitted(self)
        X = check_array(X)
        
        predictions = np.zeros(len(X), dtype=int)
        
        for i in range(len(X)):
            bmu_idx = self._find_bmu(X[i])
            predictions[i] = self.classes_[self.neuron_labels_[bmu_idx]]
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data
        
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self)
        X = check_array(X)
        
        probas = np.zeros((len(X), len(self.classes_)))
        
        for i in range(len(X)):
            bmu_idx = self._find_bmu(X[i])
            counts = self.neuron_class_counts_[bmu_idx]
            if counts.sum() > 0:
                probas[i] = counts / counts.sum()
            else:
                probas[i] = np.ones(len(self.classes_)) / len(self.classes_)
        
        return probas