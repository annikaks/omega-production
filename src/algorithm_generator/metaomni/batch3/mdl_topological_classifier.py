import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.spatial.distance import cdist
from scipy.stats import entropy


class MDLTopologicalClassifier(BaseEstimator, ClassifierMixin):
    """
    Compress persistence diagrams by encoding only topological features that exceed
    minimum description length (MDL) thresholds, filtering noise through 
    information-theoretic criteria.
    
    Parameters
    ----------
    mdl_threshold : float, default=1.0
        Minimum description length threshold for feature selection.
        Higher values result in more aggressive compression.
    
    persistence_threshold : float, default=0.1
        Minimum persistence (death - birth) for topological features.
        Features with lower persistence are considered noise.
    
    n_bins : int, default=20
        Number of bins for histogram-based feature encoding.
    
    metric : str, default='euclidean'
        Distance metric for nearest neighbor classification.
    """
    
    def __init__(self, mdl_threshold=1.0, persistence_threshold=0.1, 
                 n_bins=20, metric='euclidean'):
        self.mdl_threshold = mdl_threshold
        self.persistence_threshold = persistence_threshold
        self.n_bins = n_bins
        self.metric = metric
    
    def _compute_persistence(self, diagram):
        """Compute persistence (death - birth) for each point in diagram."""
        if diagram.ndim == 1:
            diagram = diagram.reshape(-1, 2)
        return diagram[:, 1] - diagram[:, 0]
    
    def _filter_by_persistence(self, diagram):
        """Filter diagram points by minimum persistence threshold."""
        if diagram.ndim == 1:
            diagram = diagram.reshape(-1, 2)
        
        if len(diagram) == 0:
            return diagram
        
        persistence = self._compute_persistence(diagram)
        mask = persistence >= self.persistence_threshold
        return diagram[mask]
    
    def _compute_mdl_score(self, feature_values):
        """
        Compute MDL score for a feature based on information-theoretic criteria.
        Lower MDL indicates more informative features.
        """
        if len(feature_values) == 0:
            return np.inf
        
        # Model complexity: number of parameters needed to encode the feature
        n_samples = len(feature_values)
        
        # Data encoding length: based on entropy
        hist, _ = np.histogram(feature_values, bins=self.n_bins, density=True)
        hist = hist + 1e-10  # Avoid log(0)
        hist = hist / hist.sum()
        data_entropy = entropy(hist)
        
        # MDL = model complexity + data encoding length
        model_complexity = np.log2(n_samples + 1)
        data_length = data_entropy * n_samples
        
        mdl = model_complexity + data_length
        return mdl
    
    def _extract_topological_features(self, diagram):
        """
        Extract compressed topological features from persistence diagram.
        Uses MDL criterion to select informative features.
        """
        if diagram.ndim == 1:
            diagram = diagram.reshape(-1, 2)
        
        # Filter by persistence threshold
        filtered_diagram = self._filter_by_persistence(diagram)
        
        if len(filtered_diagram) == 0:
            return np.zeros(6)  # Return zero features if no points survive
        
        # Compute candidate features
        births = filtered_diagram[:, 0]
        deaths = filtered_diagram[:, 1]
        persistence = self._compute_persistence(filtered_diagram)
        midpoints = (births + deaths) / 2
        
        features = []
        feature_candidates = [
            ('mean_birth', np.mean(births)),
            ('std_birth', np.std(births)),
            ('mean_death', np.mean(deaths)),
            ('std_death', np.std(deaths)),
            ('mean_persistence', np.mean(persistence)),
            ('max_persistence', np.max(persistence)),
            ('sum_persistence', np.sum(persistence)),
            ('n_features', len(filtered_diagram)),
            ('mean_midpoint', np.mean(midpoints)),
            ('std_midpoint', np.std(midpoints)),
        ]
        
        # Compute MDL scores for each feature type across training data
        selected_features = []
        for name, value in feature_candidates:
            selected_features.append(value)
        
        return np.array(selected_features)
    
    def _encode_diagram(self, diagram):
        """
        Encode a persistence diagram into a compressed feature vector
        using MDL-based feature selection.
        """
        if isinstance(diagram, list):
            # Handle multiple homology dimensions
            all_features = []
            for dim_diagram in diagram:
                features = self._extract_topological_features(dim_diagram)
                all_features.append(features)
            return np.concatenate(all_features)
        else:
            return self._extract_topological_features(diagram)
    
    def _process_input(self, X):
        """Process input to handle various persistence diagram formats."""
        if isinstance(X, list):
            # List of diagrams
            return np.array([self._encode_diagram(diagram) for diagram in X])
        elif isinstance(X, np.ndarray):
            if X.ndim == 2:
                # Single diagram or already encoded features
                if X.shape[1] == 2:
                    # Single diagram with (birth, death) pairs
                    return self._encode_diagram(X).reshape(1, -1)
                else:
                    # Already encoded features
                    return X
            elif X.ndim == 3:
                # Multiple diagrams
                return np.array([self._encode_diagram(X[i]) for i in range(len(X))])
        
        return X
    
    def _select_features_by_mdl(self, X, y):
        """Select features that meet MDL threshold criterion."""
        n_features = X.shape[1]
        selected_indices = []
        
        for i in range(n_features):
            feature_values = X[:, i]
            
            # Compute MDL for this feature
            mdl_score = self._compute_mdl_score(feature_values)
            
            # Compute discriminative power (class separation)
            class_means = []
            for label in self.classes_:
                class_mask = y == label
                if np.sum(class_mask) > 0:
                    class_means.append(np.mean(feature_values[class_mask]))
            
            if len(class_means) > 1:
                discriminative_power = np.std(class_means)
            else:
                discriminative_power = 0
            
            # Select feature if MDL is low enough and discriminative power is high
            if mdl_score < self.mdl_threshold * 100 and discriminative_power > 1e-6:
                selected_indices.append(i)
        
        # Ensure at least one feature is selected
        if len(selected_indices) == 0:
            selected_indices = [0]
        
        return np.array(selected_indices)
    
    def fit(self, X_train, y_train):
        """
        Fit the MDL topological classifier.
        
        Parameters
        ----------
        X_train : array-like or list of persistence diagrams
            Training data.
        y_train : array-like of shape (n_samples,)
            Target labels.
        
        Returns
        -------
        self : object
            Fitted classifier.
        """
        # Process input diagrams
        X_encoded = self._process_input(X_train)
        
        # Validate input
        X_encoded, y_train = check_X_y(X_encoded, y_train)
        
        # Store classes
        self.classes_ = unique_labels(y_train)
        
        # Select features based on MDL criterion
        self.selected_features_ = self._select_features_by_mdl(X_encoded, y_train)
        
        # Store compressed training data
        self.X_train_ = X_encoded[:, self.selected_features_]
        self.y_train_ = y_train
        
        # Compute class prototypes for efficient prediction
        self.class_prototypes_ = {}
        for label in self.classes_:
            class_mask = y_train == label
            self.class_prototypes_[label] = np.mean(
                self.X_train_[class_mask], axis=0
            )
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels for test data.
        
        Parameters
        ----------
        X_test : array-like or list of persistence diagrams
            Test data.
        
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fitted
        check_is_fitted(self, ['X_train_', 'y_train_', 'selected_features_'])
        
        # Process input diagrams
        X_encoded = self._process_input(X_test)
        
        # Validate input
        X_encoded = check_array(X_encoded)
        
        # Select features
        X_test_compressed = X_encoded[:, self.selected_features_]
        
        # Nearest neighbor classification using class prototypes
        predictions = []
        for x in X_test_compressed:
            min_dist = np.inf
            pred_label = self.classes_[0]
            
            for label, prototype in self.class_prototypes_.items():
                dist = np.linalg.norm(x - prototype)
                if dist < min_dist:
                    min_dist = dist
                    pred_label = label
            
            predictions.append(pred_label)
        
        return np.array(predictions)
    
    def get_compression_ratio(self):
        """
        Get the compression ratio achieved by MDL-based feature selection.
        
        Returns
        -------
        ratio : float
            Ratio of selected features to total features.
        """
        check_is_fitted(self, ['selected_features_'])
        total_features = len(self.selected_features_) / (
            len(self.selected_features_) / self.X_train_.shape[1]
        )
        return len(self.selected_features_) / self.X_train_.shape[1]