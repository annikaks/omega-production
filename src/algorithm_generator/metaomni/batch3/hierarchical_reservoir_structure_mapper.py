import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix
from scipy.linalg import eigh


class HierarchicalReservoirStructureMapper(BaseEstimator, ClassifierMixin):
    """
    Hierarchical Reservoir Computing with discrete symbolic transitions between layers
    and continuous dynamics within layers for structure mapping classification.
    
    Parameters
    ----------
    n_layers : int, default=3
        Number of hierarchical reservoir layers
    reservoir_sizes : list of int, default=None
        Size of each reservoir layer. If None, uses [100, 50, 25]
    spectral_radius : float, default=0.9
        Spectral radius for reservoir weight matrices
    input_scaling : float, default=0.5
        Scaling factor for input weights
    leak_rate : float, default=0.3
        Leak rate for leaky integrator neurons
    sparsity : float, default=0.1
        Sparsity of reservoir connections
    n_symbols : int, default=10
        Number of discrete symbols for inter-layer transitions
    symbol_threshold : str, default='quantile'
        Method for symbol discretization ('quantile' or 'uniform')
    ridge_alpha : float, default=1.0
        Regularization parameter for ridge regression readout
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_layers=3, reservoir_sizes=None, spectral_radius=0.9,
                 input_scaling=0.5, leak_rate=0.3, sparsity=0.1, n_symbols=10,
                 symbol_threshold='quantile', ridge_alpha=1.0, random_state=None):
        self.n_layers = n_layers
        self.reservoir_sizes = reservoir_sizes if reservoir_sizes else [100, 50, 25]
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.sparsity = sparsity
        self.n_symbols = n_symbols
        self.symbol_threshold = symbol_threshold
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        
    def _initialize_reservoir(self, input_dim, reservoir_dim):
        """Initialize reservoir weights with specified spectral radius."""
        rng = np.random.RandomState(self.random_state)
        
        # Input weights
        W_in = rng.uniform(-self.input_scaling, self.input_scaling, 
                          (reservoir_dim, input_dim))
        
        # Reservoir weights (sparse)
        W_res = rng.randn(reservoir_dim, reservoir_dim)
        mask = rng.rand(reservoir_dim, reservoir_dim) < self.sparsity
        W_res = W_res * mask
        
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W_res)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            W_res = W_res * (self.spectral_radius / current_radius)
        
        return W_in, W_res
    
    def _reservoir_dynamics(self, inputs, W_in, W_res, initial_state=None):
        """Compute continuous reservoir dynamics for a sequence."""
        n_samples, n_timesteps, n_features = inputs.shape
        reservoir_dim = W_res.shape[0]
        
        states = np.zeros((n_samples, n_timesteps, reservoir_dim))
        
        for i in range(n_samples):
            state = initial_state[i] if initial_state is not None else np.zeros(reservoir_dim)
            
            for t in range(n_timesteps):
                # Leaky integrator dynamics
                pre_activation = (W_in @ inputs[i, t] + W_res @ state)
                state = (1 - self.leak_rate) * state + self.leak_rate * np.tanh(pre_activation)
                states[i, t] = state
        
        return states
    
    def _discretize_to_symbols(self, continuous_states):
        """Convert continuous states to discrete symbols."""
        n_samples, n_timesteps, n_features = continuous_states.shape
        symbols = np.zeros((n_samples, n_timesteps, n_features), dtype=int)
        
        for f in range(n_features):
            feature_data = continuous_states[:, :, f].flatten()
            
            if self.symbol_threshold == 'quantile':
                # Quantile-based discretization
                thresholds = np.percentile(feature_data, 
                                          np.linspace(0, 100, self.n_symbols + 1))
            else:
                # Uniform discretization
                min_val, max_val = feature_data.min(), feature_data.max()
                thresholds = np.linspace(min_val, max_val, self.n_symbols + 1)
            
            # Assign symbols
            for s in range(self.n_symbols):
                mask = (feature_data >= thresholds[s]) & (feature_data < thresholds[s + 1])
                symbols[:, :, f].flat[mask] = s
            
            # Handle edge case for maximum value
            symbols[:, :, f].flat[feature_data >= thresholds[-1]] = self.n_symbols - 1
        
        return symbols
    
    def _symbol_to_continuous(self, symbols, symbol_dim):
        """Convert discrete symbols to continuous representation for next layer."""
        n_samples, n_timesteps, n_features = symbols.shape
        
        # One-hot encode symbols and flatten
        continuous = np.zeros((n_samples, n_timesteps, symbol_dim))
        
        for i in range(n_samples):
            for t in range(n_timesteps):
                # Create feature vector from symbol combination
                symbol_vec = symbols[i, t]
                # Hash-based continuous representation
                feature_idx = 0
                for s_idx, s_val in enumerate(symbol_vec):
                    hash_val = (s_val + s_idx * self.n_symbols) % symbol_dim
                    continuous[i, t, hash_val] += 1.0
                
                # Normalize
                norm = np.linalg.norm(continuous[i, t])
                if norm > 0:
                    continuous[i, t] /= norm
        
        return continuous
    
    def _prepare_input(self, X):
        """Prepare input data as 3D array (samples, timesteps, features)."""
        if X.ndim == 2:
            # Treat each sample as a single timestep
            X = X[:, np.newaxis, :]
        elif X.ndim == 1:
            X = X[np.newaxis, np.newaxis, :]
        return X
    
    def fit(self, X_train, y_train):
        """
        Fit the hierarchical reservoir structure mapper.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features) or (n_samples, n_timesteps, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target labels
            
        Returns
        -------
        self : object
            Fitted estimator
        """
        rng = np.random.RandomState(self.random_state)
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y_train)
        self.classes_ = self.label_encoder_.classes_
        
        # Prepare input
        X_train = self._prepare_input(X_train)
        n_samples, n_timesteps, n_features = X_train.shape
        
        # Ensure reservoir sizes match number of layers
        if len(self.reservoir_sizes) != self.n_layers:
            self.reservoir_sizes = [max(50, 100 // (i + 1)) for i in range(self.n_layers)]
        
        # Initialize reservoirs for each layer
        self.reservoirs_ = []
        self.symbol_dims_ = []
        
        current_input = X_train
        current_input_dim = n_features
        all_layer_states = []
        
        for layer_idx in range(self.n_layers):
            reservoir_dim = self.reservoir_sizes[layer_idx]
            
            # Initialize reservoir
            W_in, W_res = self._initialize_reservoir(current_input_dim, reservoir_dim)
            
            # Compute continuous dynamics
            states = self._reservoir_dynamics(current_input, W_in, W_res)
            all_layer_states.append(states)
            
            # Store reservoir
            self.reservoirs_.append({
                'W_in': W_in,
                'W_res': W_res,
                'layer_idx': layer_idx
            })
            
            # Discretize to symbols for next layer (except last layer)
            if layer_idx < self.n_layers - 1:
                symbols = self._discretize_to_symbols(states)
                
                # Convert symbols to continuous for next layer
                symbol_dim = min(reservoir_dim, self.n_symbols * states.shape[2])
                self.symbol_dims_.append(symbol_dim)
                current_input = self._symbol_to_continuous(symbols, symbol_dim)
                current_input_dim = symbol_dim
        
        # Aggregate features from all layers
        aggregated_features = []
        for states in all_layer_states:
            # Use mean and max pooling over time
            mean_states = np.mean(states, axis=1)
            max_states = np.max(states, axis=1)
            aggregated_features.append(mean_states)
            aggregated_features.append(max_states)
        
        X_combined = np.hstack(aggregated_features)
        
        # Train readout layer with Ridge regression
        self.readout_ = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
        
        # One-hot encode targets for multi-class
        n_classes = len(self.classes_)
        y_onehot = np.zeros((len(y_encoded), n_classes))
        y_onehot[np.arange(len(y_encoded)), y_encoded] = 1
        
        self.readout_.fit(X_combined, y_onehot)
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features) or (n_samples, n_timesteps, n_features)
            Test data
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        # Prepare input
        X_test = self._prepare_input(X_test)
        
        current_input = X_test
        all_layer_states = []
        
        for layer_idx in range(self.n_layers):
            reservoir = self.reservoirs_[layer_idx]
            W_in = reservoir['W_in']
            W_res = reservoir['W_res']
            
            # Compute continuous dynamics
            states = self._reservoir_dynamics(current_input, W_in, W_res)
            all_layer_states.append(states)
            
            # Discretize to symbols for next layer (except last layer)
            if layer_idx < self.n_layers - 1:
                symbols = self._discretize_to_symbols(states)
                symbol_dim = self.symbol_dims_[layer_idx]
                current_input = self._symbol_to_continuous(symbols, symbol_dim)
        
        # Aggregate features from all layers
        aggregated_features = []
        for states in all_layer_states:
            mean_states = np.mean(states, axis=1)
            max_states = np.max(states, axis=1)
            aggregated_features.append(mean_states)
            aggregated_features.append(max_states)
        
        X_combined = np.hstack(aggregated_features)
        
        # Predict using readout layer
        y_pred_proba = self.readout_.predict(X_combined)
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)
        
        return self.label_encoder_.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features) or (n_samples, n_timesteps, n_features)
            Test data
            
        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        X_test = self._prepare_input(X_test)
        
        current_input = X_test
        all_layer_states = []
        
        for layer_idx in range(self.n_layers):
            reservoir = self.reservoirs_[layer_idx]
            W_in = reservoir['W_in']
            W_res = reservoir['W_res']
            
            states = self._reservoir_dynamics(current_input, W_in, W_res)
            all_layer_states.append(states)
            
            if layer_idx < self.n_layers - 1:
                symbols = self._discretize_to_symbols(states)
                symbol_dim = self.symbol_dims_[layer_idx]
                current_input = self._symbol_to_continuous(symbols, symbol_dim)
        
        aggregated_features = []
        for states in all_layer_states:
            mean_states = np.mean(states, axis=1)
            max_states = np.max(states, axis=1)
            aggregated_features.append(mean_states)
            aggregated_features.append(max_states)
        
        X_combined = np.hstack(aggregated_features)
        
        y_pred_proba = self.readout_.predict(X_combined)
        
        # Normalize to probabilities
        y_pred_proba = np.clip(y_pred_proba, 0, 1)
        row_sums = y_pred_proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        y_pred_proba = y_pred_proba / row_sums
        
        return y_pred_proba