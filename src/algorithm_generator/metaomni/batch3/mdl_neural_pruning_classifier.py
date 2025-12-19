import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class MDLNeuralPruningClassifier(BaseEstimator, ClassifierMixin):
    """
    Neural network classifier with Minimum Description Length (MDL) based pruning.
    
    This classifier trains a neural network and prunes redundant neurons by
    measuring the trade-off between model complexity (codebook size) and
    reconstruction error using the MDL principle.
    
    Parameters
    ----------
    hidden_layer_sizes : tuple, default=(100,)
        Initial hidden layer sizes before pruning.
    
    pruning_threshold : float, default=0.01
        Threshold for neuron importance. Neurons with importance below this
        are candidates for pruning.
    
    mdl_alpha : float, default=1.0
        Weight for the model complexity term in MDL calculation.
    
    max_iter : int, default=200
        Maximum number of iterations for training.
    
    random_state : int, default=None
        Random state for reproducibility.
    
    learning_rate_init : float, default=0.001
        Initial learning rate for training.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels.
    
    model_ : MLPClassifier
        The trained neural network model.
    
    pruned_neurons_ : list
        Indices of neurons that were pruned in each layer.
    
    mdl_score_ : float
        Final MDL score after pruning.
    """
    
    def __init__(self, hidden_layer_sizes=(100,), pruning_threshold=0.01,
                 mdl_alpha=1.0, max_iter=200, random_state=None,
                 learning_rate_init=0.001):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.pruning_threshold = pruning_threshold
        self.mdl_alpha = mdl_alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.learning_rate_init = learning_rate_init
    
    def _calculate_mdl(self, model, X, y):
        """
        Calculate the Minimum Description Length score.
        
        MDL = L(Model) + L(Data|Model)
        where L(Model) is the model complexity and L(Data|Model) is the
        reconstruction error.
        """
        # Model complexity: number of parameters
        n_params = sum(w.size for w in model.coefs_)
        model_complexity = n_params * np.log2(X.shape[0])
        
        # Data encoding length: reconstruction error
        y_pred = model.predict(X)
        error_rate = np.mean(y_pred != y)
        data_length = -X.shape[0] * np.log2(max(1 - error_rate, 1e-10))
        
        mdl_score = self.mdl_alpha * model_complexity + data_length
        return mdl_score
    
    def _compute_neuron_importance(self, model, X, y):
        """
        Compute importance of each neuron based on weight magnitudes and
        activation patterns.
        """
        importances = []
        
        for layer_idx, (coef, intercept) in enumerate(zip(model.coefs_[:-1], 
                                                           model.intercepts_[:-1])):
            # Compute importance based on incoming and outgoing weights
            incoming_weights = np.abs(coef).sum(axis=0)
            
            if layer_idx < len(model.coefs_) - 1:
                outgoing_weights = np.abs(model.coefs_[layer_idx + 1]).sum(axis=1)
            else:
                outgoing_weights = np.ones(coef.shape[1])
            
            # Combined importance score
            neuron_importance = incoming_weights * outgoing_weights
            importances.append(neuron_importance)
        
        return importances
    
    def _prune_neurons(self, model, importances):
        """
        Prune neurons with low importance scores.
        """
        pruned_indices = []
        
        for layer_idx, importance in enumerate(importances):
            # Normalize importance scores
            if importance.max() > 0:
                normalized_importance = importance / importance.max()
            else:
                normalized_importance = importance
            
            # Identify neurons to keep
            keep_mask = normalized_importance >= self.pruning_threshold
            
            # Ensure at least one neuron remains
            if keep_mask.sum() == 0:
                keep_mask[np.argmax(normalized_importance)] = True
            
            pruned_idx = np.where(~keep_mask)[0]
            pruned_indices.append(pruned_idx)
        
        return pruned_indices
    
    def _create_pruned_model(self, original_model, pruned_indices):
        """
        Create a new model with pruned architecture.
        """
        # Calculate new hidden layer sizes
        new_hidden_sizes = []
        for layer_idx, (original_size, pruned_idx) in enumerate(
                zip(self.hidden_layer_sizes, pruned_indices)):
            new_size = original_size - len(pruned_idx)
            new_hidden_sizes.append(max(new_size, 1))  # At least 1 neuron
        
        # Create new model with pruned architecture
        pruned_model = MLPClassifier(
            hidden_layer_sizes=tuple(new_hidden_sizes),
            max_iter=self.max_iter,
            random_state=self.random_state,
            learning_rate_init=self.learning_rate_init,
            warm_start=False
        )
        
        return pruned_model, new_hidden_sizes
    
    def _transfer_weights(self, source_model, target_model, pruned_indices, X, y):
        """
        Transfer weights from source to target model, excluding pruned neurons.
        """
        # Train target model first to initialize structure
        target_model.fit(X, y)
        
        # Transfer weights layer by layer
        for layer_idx in range(len(source_model.coefs_)):
            if layer_idx < len(pruned_indices):
                # Create mask for neurons to keep
                keep_mask = np.ones(source_model.coefs_[layer_idx].shape[1], 
                                   dtype=bool)
                keep_mask[pruned_indices[layer_idx]] = False
                
                # Transfer weights for kept neurons
                if layer_idx < len(target_model.coefs_):
                    source_coef = source_model.coefs_[layer_idx][:, keep_mask]
                    target_shape = target_model.coefs_[layer_idx].shape
                    
                    if source_coef.shape == target_shape:
                        target_model.coefs_[layer_idx] = source_coef.copy()
                        target_model.intercepts_[layer_idx] = \
                            source_model.intercepts_[layer_idx][keep_mask].copy()
            else:
                # Output layer - adjust for previous pruning
                if layer_idx > 0 and layer_idx - 1 < len(pruned_indices):
                    keep_mask = np.ones(source_model.coefs_[layer_idx].shape[0], 
                                       dtype=bool)
                    keep_mask[pruned_indices[layer_idx - 1]] = False
                    
                    source_coef = source_model.coefs_[layer_idx][keep_mask, :]
                    target_shape = target_model.coefs_[layer_idx].shape
                    
                    if source_coef.shape == target_shape:
                        target_model.coefs_[layer_idx] = source_coef.copy()
        
        return target_model
    
    def fit(self, X, y):
        """
        Fit the MDL-based pruned neural network classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y : array-like of shape (n_samples,)
            Target values.
        
        Returns
        -------
        self : object
            Returns self.
        """
        # Validate input
        X, y = check_X_y(X, y)
        
        # Store classes
        self.classes_ = unique_labels(y)
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        
        # Train initial model
        self.model_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=self.random_state,
            learning_rate_init=self.learning_rate_init
        )
        self.model_.fit(X, y_encoded)
        
        # Calculate initial MDL
        initial_mdl = self._calculate_mdl(self.model_, X, y_encoded)
        
        # Compute neuron importance
        importances = self._compute_neuron_importance(self.model_, X, y_encoded)
        
        # Prune neurons
        self.pruned_neurons_ = self._prune_neurons(self.model_, importances)
        
        # Create and train pruned model
        pruned_model, new_hidden_sizes = self._create_pruned_model(
            self.model_, self.pruned_neurons_)
        
        # Train pruned model
        pruned_model.fit(X, y_encoded)
        
        # Calculate MDL after pruning
        pruned_mdl = self._calculate_mdl(pruned_model, X, y_encoded)
        
        # Use pruned model if it has better MDL score
        if pruned_mdl < initial_mdl:
            self.model_ = pruned_model
            self.mdl_score_ = pruned_mdl
            self.final_hidden_sizes_ = new_hidden_sizes
        else:
            self.mdl_score_ = initial_mdl
            self.final_hidden_sizes_ = self.hidden_layer_sizes
            self.pruned_neurons_ = [np.array([]) for _ in self.hidden_layer_sizes]
        
        self.n_features_in_ = X.shape[1]
        
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
        check_is_fitted(self, ['model_', 'classes_'])
        X = check_array(X)
        
        y_pred_encoded = self.model_.predict(X)
        y_pred = self.label_encoder_.inverse_transform(y_pred_encoded)
        
        return y_pred
    
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
            Predicted class probabilities.
        """
        check_is_fitted(self, ['model_', 'classes_'])
        X = check_array(X)
        
        return self.model_.predict_proba(X)
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        
        y : array-like of shape (n_samples,)
            True labels.
        
        Returns
        -------
        score : float
            Mean accuracy.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'pruning_threshold': self.pruning_threshold,
            'mdl_alpha': self.mdl_alpha,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'learning_rate_init': self.learning_rate_init
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self