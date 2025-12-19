import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import mode


class HistogramEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Bootstrap ensemble classifier using histogram-based binning strategies.
    
    Each sub-model in the ensemble uses a different random binning strategy
    to discretize features before making predictions.
    
    Parameters
    ----------
    n_estimators : int, default=10
        Number of sub-models in the ensemble.
    min_bins : int, default=3
        Minimum number of bins for histogram discretization.
    max_bins : int, default=20
        Maximum number of bins for histogram discretization.
    bootstrap_sample_ratio : float, default=1.0
        Ratio of samples to use for each bootstrap sample.
    random_state : int, default=None
        Random seed for reproducibility.
    """
    
    def __init__(self, n_estimators=10, min_bins=3, max_bins=20, 
                 bootstrap_sample_ratio=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.bootstrap_sample_ratio = bootstrap_sample_ratio
        self.random_state = random_state
    
    def _create_binning_strategy(self, n_features, rng):
        """Create random binning strategy for each feature."""
        strategy = {}
        for feature_idx in range(n_features):
            n_bins = rng.randint(self.min_bins, self.max_bins + 1)
            strategy[feature_idx] = n_bins
        return strategy
    
    def _discretize_features(self, X, bin_edges):
        """Discretize features using precomputed bin edges."""
        X_discretized = np.zeros_like(X, dtype=np.int32)
        for feature_idx in range(X.shape[1]):
            edges = bin_edges[feature_idx]
            X_discretized[:, feature_idx] = np.digitize(X[:, feature_idx], edges[1:-1])
        return X_discretized
    
    def _compute_bin_edges(self, X, binning_strategy):
        """Compute bin edges for each feature based on training data."""
        bin_edges = {}
        for feature_idx, n_bins in binning_strategy.items():
            feature_data = X[:, feature_idx]
            edges = np.histogram_bin_edges(feature_data, bins=n_bins)
            bin_edges[feature_idx] = edges
        return bin_edges
    
    def _build_histogram_model(self, X_discretized, y):
        """Build histogram-based probability model."""
        n_features = X_discretized.shape[1]
        model = {}
        
        for class_label in self.classes_:
            class_mask = (y == class_label)
            X_class = X_discretized[class_mask]
            
            feature_histograms = {}
            for feature_idx in range(n_features):
                feature_values = X_class[:, feature_idx]
                max_bin = X_discretized[:, feature_idx].max()
                
                # Count occurrences with Laplace smoothing
                counts = np.bincount(feature_values, minlength=max_bin + 1)
                probabilities = (counts + 1) / (len(feature_values) + max_bin + 1)
                feature_histograms[feature_idx] = probabilities
            
            model[class_label] = feature_histograms
        
        return model
    
    def _predict_single_model(self, X_discretized, histogram_model, class_prior):
        """Make predictions using a single histogram model."""
        n_samples = X_discretized.shape[0]
        n_classes = len(self.classes_)
        log_probs = np.zeros((n_samples, n_classes))
        
        for class_idx, class_label in enumerate(self.classes_):
            feature_histograms = histogram_model[class_label]
            log_prob = np.log(class_prior[class_label])
            
            for feature_idx, probabilities in feature_histograms.items():
                feature_values = X_discretized[:, feature_idx]
                # Handle out-of-range bins
                feature_values = np.clip(feature_values, 0, len(probabilities) - 1)
                log_prob += np.log(probabilities[feature_values])
            
            log_probs[:, class_idx] = log_prob
        
        return self.classes_[np.argmax(log_probs, axis=1)]
    
    def fit(self, X_train, y_train):
        """
        Fit the histogram ensemble classifier.
        
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
        X_train, y_train = check_X_y(X_train, y_train)
        self.classes_ = unique_labels(y_train)
        self.n_features_in_ = X_train.shape[1]
        
        rng = np.random.RandomState(self.random_state)
        
        self.models_ = []
        n_samples = X_train.shape[0]
        bootstrap_size = int(n_samples * self.bootstrap_sample_ratio)
        
        for estimator_idx in range(self.n_estimators):
            # Bootstrap sampling
            bootstrap_indices = rng.choice(n_samples, size=bootstrap_size, replace=True)
            X_bootstrap = X_train[bootstrap_indices]
            y_bootstrap = y_train[bootstrap_indices]
            
            # Create random binning strategy
            binning_strategy = self._create_binning_strategy(self.n_features_in_, rng)
            
            # Compute bin edges from bootstrap sample
            bin_edges = self._compute_bin_edges(X_bootstrap, binning_strategy)
            
            # Discretize features
            X_discretized = self._discretize_features(X_bootstrap, bin_edges)
            
            # Build histogram model
            histogram_model = self._build_histogram_model(X_discretized, y_bootstrap)
            
            # Compute class prior
            class_prior = {}
            for class_label in self.classes_:
                class_prior[class_label] = np.mean(y_bootstrap == class_label)
            
            # Store model components
            self.models_.append({
                'binning_strategy': binning_strategy,
                'bin_edges': bin_edges,
                'histogram_model': histogram_model,
                'class_prior': class_prior
            })
        
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
        check_is_fitted(self, ['models_', 'classes_'])
        X_test = check_array(X_test)
        
        if X_test.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_test.shape[1]}")
        
        # Collect predictions from all models
        all_predictions = []
        
        for model_components in self.models_:
            bin_edges = model_components['bin_edges']
            histogram_model = model_components['histogram_model']
            class_prior = model_components['class_prior']
            
            # Discretize test data
            X_discretized = self._discretize_features(X_test, bin_edges)
            
            # Make predictions
            predictions = self._predict_single_model(X_discretized, histogram_model, class_prior)
            all_predictions.append(predictions)
        
        # Aggregate predictions by majority voting
        all_predictions = np.array(all_predictions)
        final_predictions = mode(all_predictions, axis=0, keepdims=False)[0]
        
        return final_predictions
    
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
        check_is_fitted(self, ['models_', 'classes_'])
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        n_classes = len(self.classes_)
        proba_sum = np.zeros((n_samples, n_classes))
        
        for model_components in self.models_:
            bin_edges = model_components['bin_edges']
            histogram_model = model_components['histogram_model']
            class_prior = model_components['class_prior']
            
            # Discretize test data
            X_discretized = self._discretize_features(X_test, bin_edges)
            
            # Compute probabilities
            log_probs = np.zeros((n_samples, n_classes))
            
            for class_idx, class_label in enumerate(self.classes_):
                feature_histograms = histogram_model[class_label]
                log_prob = np.log(class_prior[class_label])
                
                for feature_idx, probabilities in feature_histograms.items():
                    feature_values = X_discretized[:, feature_idx]
                    feature_values = np.clip(feature_values, 0, len(probabilities) - 1)
                    log_prob += np.log(probabilities[feature_values])
                
                log_probs[:, class_idx] = log_prob
            
            # Convert log probabilities to probabilities
            probs = np.exp(log_probs - log_probs.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            proba_sum += probs
        
        # Average probabilities across all models
        return proba_sum / self.n_estimators