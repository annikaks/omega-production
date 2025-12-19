import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import KBinsDiscretizer


class StochasticHistogramClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that uses stochastic histogram merging for improved generalization.
    
    During training, adjacent histogram bins are randomly combined to inject controlled
    noise, which helps prevent overfitting and improves generalization.
    
    Parameters
    ----------
    n_bins : int, default=10
        Number of initial bins for histogram discretization.
    merge_probability : float, default=0.3
        Probability of merging adjacent bins during training (0.0 to 1.0).
    n_merge_iterations : int, default=5
        Number of stochastic merging iterations during training.
    strategy : str, default='uniform'
        Strategy for initial binning ('uniform', 'quantile', 'kmeans').
    random_state : int or None, default=None
        Random seed for reproducibility.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes seen during fit.
    n_features_in_ : int
        Number of features seen during fit.
    discretizers_ : list
        List of fitted KBinsDiscretizer objects for each feature.
    histograms_ : dict
        Dictionary mapping (feature_idx, bin_idx) to class probabilities.
    """
    
    def __init__(self, n_bins=10, merge_probability=0.3, n_merge_iterations=5,
                 strategy='uniform', random_state=None):
        self.n_bins = n_bins
        self.merge_probability = merge_probability
        self.n_merge_iterations = n_merge_iterations
        self.strategy = strategy
        self.random_state = random_state
    
    def _merge_bins_stochastically(self, bin_assignments, rng):
        """
        Randomly merge adjacent bins based on merge_probability.
        
        Parameters
        ----------
        bin_assignments : ndarray
            Array of bin assignments for samples.
        rng : np.random.Generator
            Random number generator.
            
        Returns
        -------
        merged_bins : ndarray
            New bin assignments after merging.
        """
        unique_bins = np.unique(bin_assignments)
        n_unique = len(unique_bins)
        
        if n_unique <= 1:
            return bin_assignments
        
        # Create merge mapping
        merge_map = {bin_id: bin_id for bin_id in unique_bins}
        
        # Sort bins to ensure adjacency
        sorted_bins = sorted(unique_bins)
        
        # Randomly decide which adjacent bins to merge
        i = 0
        while i < len(sorted_bins) - 1:
            if rng.random() < self.merge_probability:
                # Merge bin i+1 into bin i
                merge_map[sorted_bins[i + 1]] = merge_map[sorted_bins[i]]
                i += 2  # Skip next bin since it's merged
            else:
                i += 1
        
        # Apply merge mapping
        merged_bins = np.array([merge_map[b] for b in bin_assignments])
        
        # Renumber bins to be consecutive
        unique_merged = np.unique(merged_bins)
        renumber_map = {old: new for new, old in enumerate(unique_merged)}
        merged_bins = np.array([renumber_map[b] for b in merged_bins])
        
        return merged_bins
    
    def fit(self, X_train, y_train):
        """
        Fit the stochastic histogram classifier.
        
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
        
        # Store classes and feature count
        self.classes_ = unique_labels(y_train)
        self.n_features_in_ = X_train.shape[1]
        
        # Initialize random number generator
        rng = np.random.default_rng(self.random_state)
        
        # Fit discretizers for each feature
        self.discretizers_ = []
        for feature_idx in range(self.n_features_in_):
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode='ordinal',
                strategy=self.strategy
            )
            discretizer.fit(X_train[:, feature_idx].reshape(-1, 1))
            self.discretizers_.append(discretizer)
        
        # Build histograms with stochastic merging
        self.histograms_ = {}
        
        for feature_idx in range(self.n_features_in_):
            # Get initial bin assignments
            bin_assignments = self.discretizers_[feature_idx].transform(
                X_train[:, feature_idx].reshape(-1, 1)
            ).astype(int).ravel()
            
            # Perform multiple iterations of stochastic merging
            for iteration in range(self.n_merge_iterations):
                # Merge bins stochastically
                merged_bins = self._merge_bins_stochastically(bin_assignments, rng)
                
                # Compute class probabilities for each bin
                unique_bins = np.unique(merged_bins)
                
                for bin_idx in unique_bins:
                    mask = merged_bins == bin_idx
                    bin_labels = y_train[mask]
                    
                    # Count class occurrences
                    class_counts = np.zeros(len(self.classes_))
                    for class_idx, class_label in enumerate(self.classes_):
                        class_counts[class_idx] = np.sum(bin_labels == class_label)
                    
                    # Normalize to probabilities with Laplace smoothing
                    class_probs = (class_counts + 1) / (len(bin_labels) + len(self.classes_))
                    
                    # Store or update histogram (average over iterations)
                    key = (feature_idx, bin_idx, iteration)
                    self.histograms_[key] = class_probs
        
        # Aggregate histograms across iterations
        self.aggregated_histograms_ = {}
        for feature_idx in range(self.n_features_in_):
            # Get all bins across all iterations for this feature
            feature_keys = [k for k in self.histograms_.keys() if k[0] == feature_idx]
            
            # Group by bin index
            bin_groups = {}
            for key in feature_keys:
                _, bin_idx, iteration = key
                if bin_idx not in bin_groups:
                    bin_groups[bin_idx] = []
                bin_groups[bin_idx].append(self.histograms_[key])
            
            # Average probabilities across iterations
            for bin_idx, prob_list in bin_groups.items():
                self.aggregated_histograms_[(feature_idx, bin_idx)] = np.mean(prob_list, axis=0)
        
        return self
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self)
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        proba = np.zeros((n_samples, len(self.classes_)))
        
        for sample_idx in range(n_samples):
            feature_probs = []
            
            for feature_idx in range(self.n_features_in_):
                # Get bin assignment
                bin_assignment = self.discretizers_[feature_idx].transform(
                    X_test[sample_idx, feature_idx].reshape(-1, 1)
                ).astype(int).ravel()[0]
                
                # Find closest bin in aggregated histograms
                feature_bins = [k[1] for k in self.aggregated_histograms_.keys() 
                               if k[0] == feature_idx]
                
                if len(feature_bins) > 0:
                    closest_bin = min(feature_bins, key=lambda b: abs(b - bin_assignment))
                    key = (feature_idx, closest_bin)
                    
                    if key in self.aggregated_histograms_:
                        feature_probs.append(self.aggregated_histograms_[key])
                    else:
                        # Uniform prior if bin not found
                        feature_probs.append(np.ones(len(self.classes_)) / len(self.classes_))
                else:
                    feature_probs.append(np.ones(len(self.classes_)) / len(self.classes_))
            
            # Combine probabilities using naive Bayes assumption
            if feature_probs:
                combined_probs = np.prod(feature_probs, axis=0)
                proba[sample_idx] = combined_probs / np.sum(combined_probs)
            else:
                proba[sample_idx] = np.ones(len(self.classes_)) / len(self.classes_)
        
        return proba
    
    def predict(self, X_test):
        """
        Predict class labels for X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]