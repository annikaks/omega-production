import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityWeightedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Similarity-Weighted Ensemble (SWE) classifier.

    Parameters:
    -----------
    base_estimators : list of estimators
        The base estimators to be used in the ensemble.
    similarity_metric : str, default='cosine'
        The similarity metric to use. Currently only 'cosine' is supported.

    Attributes:
    -----------
    base_estimators_ : list of fitted estimators
        The fitted base estimators.
    classes_ : array-like of shape (n_classes,)
        The classes labels.
    X_ : array-like of shape (n_samples, n_features)
        The input samples used for fitting.
    y_ : array-like of shape (n_samples,)
        The target values used for fitting.
    """

    def __init__(self, base_estimators, similarity_metric='cosine'):
        self.base_estimators = base_estimators
        self.similarity_metric = similarity_metric

    def fit(self, X, y):
        """
        Fit the Similarity-Weighted Ensemble classifier.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns:
        --------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Fit all base estimators
        self.base_estimators_ = [estimator.fit(X, y) for estimator in self.base_estimators]
        
        # Store the training data
        self.X_ = X
        self.y_ = y
        
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Compute similarity between test samples and training samples
        if self.similarity_metric == 'cosine':
            similarities = cosine_similarity(X, self.X_)
        else:
            raise ValueError("Only 'cosine' similarity is currently supported.")

        # Get predictions from all base estimators
        all_predictions = np.array([estimator.predict(X) for estimator in self.base_estimators_])

        # Compute weighted predictions
        weighted_predictions = np.zeros((X.shape[0], len(self.classes_)))
        for i, pred in enumerate(all_predictions):
            for j, class_label in enumerate(self.classes_):
                mask = (pred == class_label)
                weighted_predictions[:, j] += np.sum(similarities[:, self.y_ == class_label] * mask[:, np.newaxis], axis=1)

        # Return the class with the highest weighted prediction
        return self.classes_[np.argmax(weighted_predictions, axis=1)]