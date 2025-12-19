import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class HierarchicalMixtureOfExperts(BaseEstimator, ClassifierMixin):
    """
    Hierarchical Mixture of Experts classifier using MLPs.
    
    A coarse-grain gating network predicts which fine-grain specialist networks
    to activate for each input region.
    
    Parameters
    ----------
    n_experts : int, default=4
        Number of specialist (expert) networks.
    
    gating_hidden_layers : tuple, default=(50, 25)
        Hidden layer sizes for the gating network.
    
    expert_hidden_layers : tuple, default=(100, 50)
        Hidden layer sizes for each expert network.
    
    max_iter : int, default=500
        Maximum number of iterations for MLP training.
    
    random_state : int, default=None
        Random state for reproducibility.
    
    clustering_method : str, default='kmeans'
        Method to partition input space ('kmeans' or 'random').
    
    alpha : float, default=0.0001
        L2 penalty parameter for MLPs.
    
    learning_rate_init : float, default=0.001
        Initial learning rate for MLPs.
    """
    
    def __init__(
        self,
        n_experts=4,
        gating_hidden_layers=(50, 25),
        expert_hidden_layers=(100, 50),
        max_iter=500,
        random_state=None,
        clustering_method='kmeans',
        alpha=0.0001,
        learning_rate_init=0.001
    ):
        self.n_experts = n_experts
        self.gating_hidden_layers = gating_hidden_layers
        self.expert_hidden_layers = expert_hidden_layers
        self.max_iter = max_iter
        self.random_state = random_state
        self.clustering_method = clustering_method
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
    
    def _partition_input_space(self, X, y):
        """Partition input space into regions for specialists."""
        if self.clustering_method == 'kmeans':
            self.clusterer_ = KMeans(
                n_clusters=self.n_experts,
                random_state=self.random_state,
                n_init=10
            )
            regions = self.clusterer_.fit_predict(X)
        elif self.clustering_method == 'random':
            rng = np.random.RandomState(self.random_state)
            regions = rng.randint(0, self.n_experts, size=len(X))
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        return regions
    
    def fit(self, X_train, y_train):
        """
        Fit the hierarchical mixture of experts model.
        
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
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X_train.shape[1]
        
        # Standardize features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_train)
        
        # Partition input space into regions
        regions = self._partition_input_space(X_scaled, y_train)
        
        # Train gating network to predict regions
        self.gating_network_ = MLPClassifier(
            hidden_layer_sizes=self.gating_hidden_layers,
            max_iter=self.max_iter,
            random_state=self.random_state,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.gating_network_.fit(X_scaled, regions)
        
        # Train specialist networks for each region
        self.expert_networks_ = []
        self.expert_valid_ = []
        
        for expert_idx in range(self.n_experts):
            # Get samples assigned to this expert
            expert_mask = regions == expert_idx
            
            # Check if expert has enough samples
            if np.sum(expert_mask) < 2:
                self.expert_networks_.append(None)
                self.expert_valid_.append(False)
                continue
            
            X_expert = X_scaled[expert_mask]
            y_expert = y_train[expert_mask]
            
            # Check if expert has multiple classes
            if len(np.unique(y_expert)) < 2:
                self.expert_networks_.append(None)
                self.expert_valid_.append(False)
                continue
            
            # Train expert network
            expert = MLPClassifier(
                hidden_layer_sizes=self.expert_hidden_layers,
                max_iter=self.max_iter,
                random_state=self.random_state,
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                early_stopping=True,
                validation_fraction=0.1
            )
            expert.fit(X_expert, y_expert)
            
            self.expert_networks_.append(expert)
            self.expert_valid_.append(True)
        
        # Train a fallback classifier on all data
        self.fallback_network_ = MLPClassifier(
            hidden_layer_sizes=self.expert_hidden_layers,
            max_iter=self.max_iter,
            random_state=self.random_state,
            alpha=self.alpha,
            learning_rate_init=self.learning_rate_init,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.fallback_network_.fit(X_scaled, y_train)
        
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
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_test.shape[1]} features, "
                f"but model expects {self.n_features_in_} features"
            )
        
        # Standardize features
        X_scaled = self.scaler_.transform(X_test)
        
        # Get gating network predictions (which expert to use)
        gating_probs = self.gating_network_.predict_proba(X_scaled)
        expert_assignments = self.gating_network_.predict(X_scaled)
        
        # Initialize predictions
        predictions = np.zeros(len(X_test), dtype=self.classes_.dtype)
        
        # Get predictions from each expert for their assigned samples
        for i, (x, expert_idx) in enumerate(zip(X_scaled, expert_assignments)):
            # Check if expert is valid
            if self.expert_valid_[expert_idx] and self.expert_networks_[expert_idx] is not None:
                expert = self.expert_networks_[expert_idx]
                predictions[i] = expert.predict(x.reshape(1, -1))[0]
            else:
                # Use fallback network if expert is invalid
                predictions[i] = self.fallback_network_.predict(x.reshape(1, -1))[0]
        
        return predictions
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X_test.shape[1]} features, "
                f"but model expects {self.n_features_in_} features"
            )
        
        # Standardize features
        X_scaled = self.scaler_.transform(X_test)
        
        # Get gating network predictions
        gating_probs = self.gating_network_.predict_proba(X_scaled)
        expert_assignments = self.gating_network_.predict(X_scaled)
        
        # Initialize probability matrix
        proba = np.zeros((len(X_test), self.n_classes_))
        
        # Get weighted predictions from experts
        for i, (x, expert_idx) in enumerate(zip(X_scaled, expert_assignments)):
            if self.expert_valid_[expert_idx] and self.expert_networks_[expert_idx] is not None:
                expert = self.expert_networks_[expert_idx]
                expert_proba = expert.predict_proba(x.reshape(1, -1))[0]
                
                # Ensure all classes are represented
                expert_classes = expert.classes_
                for j, cls in enumerate(self.classes_):
                    if cls in expert_classes:
                        cls_idx = np.where(expert_classes == cls)[0][0]
                        proba[i, j] = expert_proba[cls_idx]
            else:
                # Use fallback network
                fallback_proba = self.fallback_network_.predict_proba(x.reshape(1, -1))[0]
                proba[i] = fallback_proba
        
        return proba