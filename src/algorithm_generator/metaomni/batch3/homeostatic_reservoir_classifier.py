import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelBinarizer
from scipy import sparse
from scipy.sparse import linalg as sp_linalg


class HomeostaticReservoirClassifier(BaseEstimator, ClassifierMixin):
    """
    Bio-inspired reservoir computing classifier with synaptic homeostasis.
    
    This classifier implements a reservoir computing approach where connections
    undergo pruning based on synaptic homeostasis principles to maintain optimal
    information compression capacity.
    
    Parameters
    ----------
    n_reservoir : int, default=500
        Number of reservoir neurons
    spectral_radius : float, default=0.9
        Spectral radius of the reservoir weight matrix
    sparsity : float, default=0.1
        Initial sparsity of reservoir connections
    input_scaling : float, default=0.5
        Scaling of input weights
    leak_rate : float, default=0.3
        Leak rate for leaky integrator neurons
    homeostasis_rate : float, default=0.01
        Rate of homeostatic adaptation
    target_activity : float, default=0.1
        Target mean activity level for homeostasis
    pruning_threshold : float, default=0.01
        Threshold for pruning weak connections
    pruning_iterations : int, default=5
        Number of homeostatic pruning iterations during training
    ridge_alpha : float, default=1e-6
        Regularization parameter for ridge regression
    random_state : int or None, default=None
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_reservoir=500,
        spectral_radius=0.9,
        sparsity=0.1,
        input_scaling=0.5,
        leak_rate=0.3,
        homeostasis_rate=0.01,
        target_activity=0.1,
        pruning_threshold=0.01,
        pruning_iterations=5,
        ridge_alpha=1e-6,
        random_state=None
    ):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.homeostasis_rate = homeostasis_rate
        self.target_activity = target_activity
        self.pruning_threshold = pruning_threshold
        self.pruning_iterations = pruning_iterations
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        
    def _initialize_reservoir(self, n_inputs):
        """Initialize reservoir weights with sparse connectivity."""
        rng = np.random.RandomState(self.random_state)
        
        # Input weights
        self.W_in_ = rng.uniform(
            -self.input_scaling,
            self.input_scaling,
            (self.n_reservoir, n_inputs)
        )
        
        # Reservoir weights (sparse)
        n_connections = int(self.n_reservoir * self.n_reservoir * self.sparsity)
        W_reservoir = sparse.lil_matrix((self.n_reservoir, self.n_reservoir))
        
        for _ in range(n_connections):
            i = rng.randint(0, self.n_reservoir)
            j = rng.randint(0, self.n_reservoir)
            W_reservoir[i, j] = rng.randn()
        
        W_reservoir = W_reservoir.tocsr()
        
        # Scale to desired spectral radius
        eigenvalues = sp_linalg.eigs(
            W_reservoir,
            k=1,
            which='LM',
            return_eigenvectors=False,
            maxiter=1000
        )
        current_radius = np.abs(eigenvalues[0])
        
        if current_radius > 0:
            W_reservoir = W_reservoir * (self.spectral_radius / current_radius)
        
        self.W_reservoir_ = W_reservoir
        self.intrinsic_excitability_ = np.ones(self.n_reservoir)
        
    def _reservoir_activations(self, X):
        """Compute reservoir state activations for input data."""
        n_samples = X.shape[0]
        states = np.zeros((n_samples, self.n_reservoir))
        state = np.zeros(self.n_reservoir)
        
        for t in range(n_samples):
            # Leaky integrator dynamics
            pre_activation = (
                self.W_in_ @ X[t] +
                self.W_reservoir_ @ state
            )
            
            # Apply intrinsic excitability (homeostatic scaling)
            pre_activation = pre_activation * self.intrinsic_excitability_
            
            # Leaky integration with tanh activation
            state = (1 - self.leak_rate) * state + self.leak_rate * np.tanh(pre_activation)
            states[t] = state
            
        return states
    
    def _apply_homeostatic_pruning(self, X):
        """
        Apply synaptic homeostasis to prune connections and maintain
        optimal information compression.
        """
        for iteration in range(self.pruning_iterations):
            # Compute reservoir activations
            states = self._reservoir_activations(X)
            
            # Calculate mean activity per neuron
            mean_activity = np.abs(states).mean(axis=0)
            
            # Homeostatic intrinsic excitability adjustment
            activity_error = self.target_activity - mean_activity
            self.intrinsic_excitability_ += self.homeostasis_rate * activity_error
            self.intrinsic_excitability_ = np.clip(self.intrinsic_excitability_, 0.1, 10.0)
            
            # Synaptic pruning based on connection strength and usage
            W_dense = self.W_reservoir_.toarray()
            
            # Calculate connection importance (weight magnitude * post-synaptic activity)
            connection_importance = np.abs(W_dense) * mean_activity[:, np.newaxis]
            
            # Prune weak connections
            threshold = np.percentile(
                connection_importance[connection_importance > 0],
                self.pruning_threshold * 100
            )
            
            mask = (np.abs(W_dense) > 0) & (connection_importance < threshold)
            W_dense[mask] = 0
            
            # Renormalize to maintain spectral radius
            self.W_reservoir_ = sparse.csr_matrix(W_dense)
            
            if self.W_reservoir_.nnz > 0:
                try:
                    eigenvalues = sp_linalg.eigs(
                        self.W_reservoir_,
                        k=1,
                        which='LM',
                        return_eigenvectors=False,
                        maxiter=1000
                    )
                    current_radius = np.abs(eigenvalues[0])
                    
                    if current_radius > 0:
                        self.W_reservoir_ = self.W_reservoir_ * (
                            self.spectral_radius / current_radius
                        )
                except:
                    pass
    
    def _compute_information_capacity(self, states):
        """
        Estimate information compression capacity using correlation dimension.
        """
        # Use variance across neurons as a proxy for information capacity
        capacity = np.var(states, axis=0).mean()
        return capacity
    
    def fit(self, X_train, y_train):
        """
        Fit the homeostatic reservoir classifier.
        
        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target values
            
        Returns
        -------
        self : object
            Fitted classifier
        """
        X_train = np.atleast_2d(X_train)
        y_train = np.asarray(y_train)
        
        n_samples, n_features = X_train.shape
        
        # Initialize label binarizer for multi-class support
        self.label_binarizer_ = LabelBinarizer()
        y_binary = self.label_binarizer_.fit_transform(y_train)
        
        if y_binary.shape[1] == 1:
            y_binary = np.hstack([1 - y_binary, y_binary])
        
        self.classes_ = self.label_binarizer_.classes_
        
        # Initialize reservoir
        self._initialize_reservoir(n_features)
        
        # Apply homeostatic pruning
        self._apply_homeostatic_pruning(X_train)
        
        # Compute final reservoir states
        reservoir_states = self._reservoir_activations(X_train)
        
        # Train readout layer with ridge regression
        self.readout_ = Ridge(alpha=self.ridge_alpha, fit_intercept=True)
        self.readout_.fit(reservoir_states, y_binary)
        
        # Store information capacity metric
        self.information_capacity_ = self._compute_information_capacity(reservoir_states)
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels
        """
        X_test = np.atleast_2d(X_test)
        
        # Compute reservoir states
        reservoir_states = self._reservoir_activations(X_test)
        
        # Predict with readout layer
        y_pred_binary = self.readout_.predict(reservoir_states)
        
        # Convert to class labels
        if len(self.classes_) == 2:
            y_pred = self.classes_[(y_pred_binary[:, 1] > 0.5).astype(int)]
        else:
            y_pred = self.classes_[np.argmax(y_pred_binary, axis=1)]
        
        return y_pred
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        X_test = np.atleast_2d(X_test)
        
        # Compute reservoir states
        reservoir_states = self._reservoir_activations(X_test)
        
        # Predict with readout layer
        y_pred_binary = self.readout_.predict(reservoir_states)
        
        # Apply softmax for probabilities
        exp_scores = np.exp(y_pred_binary - np.max(y_pred_binary, axis=1, keepdims=True))
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return proba
    
    def get_reservoir_sparsity(self):
        """Return current sparsity of reservoir connections."""
        total_connections = self.n_reservoir * self.n_reservoir
        active_connections = self.W_reservoir_.nnz
        return 1.0 - (active_connections / total_connections)