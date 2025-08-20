import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import entropy

class EntropyGuidedDynamicNet(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=5, min_samples_split=2, n_estimators=10, entropy_threshold=0.5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.entropy_threshold = entropy_threshold
        self.estimators_ = []

    def _calculate_local_entropy(self, X, y):
        """Calculate local entropy for each sample."""
        local_entropies = []
        for i in range(X.shape[0]):
            # Find k-nearest neighbors (k=5 in this example)
            distances = np.sum((X - X[i])**2, axis=1)
            nearest_indices = np.argsort(distances)[1:6]  # Exclude the point itself
            local_labels = y[nearest_indices]
            
            # Calculate entropy of local labels
            _, counts = np.unique(local_labels, return_counts=True)
            local_entropies.append(entropy(counts))
        return np.array(local_entropies)

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        # Calculate local entropy for each sample
        local_entropies = self._calculate_local_entropy(X, y)

        # Create decision trees based on local entropy
        for _ in range(self.n_estimators):
            # Sample data points with probability proportional to their local entropy
            sample_probs = local_entropies / np.sum(local_entropies)
            sample_indices = np.random.choice(X.shape[0], size=X.shape[0], p=sample_probs)
            X_sample, y_sample = X[sample_indices], y[sample_indices]

            # Adjust tree depth based on average entropy of the sample
            avg_entropy = np.mean(local_entropies[sample_indices])
            adjusted_depth = int(self.max_depth * (1 + avg_entropy / self.entropy_threshold))

            # Create and fit a decision tree
            tree = DecisionTreeClassifier(max_depth=adjusted_depth, 
                                          min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Make predictions using all estimators
        predictions = np.array([estimator.predict(X) for estimator in self.estimators_])
        
        # Return the most common prediction for each sample
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Make probability predictions using all estimators
        probas = np.array([estimator.predict_proba(X) for estimator in self.estimators_])
        
        # Return the average probability across all estimators
        return np.mean(probas, axis=0)