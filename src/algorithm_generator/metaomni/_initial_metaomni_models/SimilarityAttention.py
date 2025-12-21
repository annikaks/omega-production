import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityAttention(BaseEstimator, ClassifierMixin):
    def __init__(self, n_attention_heads=1, similarity_threshold=0.5):
        self.n_attention_heads = n_attention_heads
        self.similarity_threshold = similarity_threshold

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_ = X.shape[1]
        
        # Initialize attention weights
        self.attention_weights_ = np.random.rand(self.n_attention_heads, self.n_features_)
        
        # Store the training data
        self.X_ = X
        self.y_ = y
        
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        # Check if X has the same number of features as the training data
        if X.shape[1] != self.n_features_:
            raise ValueError("The number of features in X does not match the number of features of the fitted data")

        # Compute similarity between test samples and training samples
        similarities = cosine_similarity(X, self.X_)

        # Apply attention mechanism
        attended_similarities = np.zeros_like(similarities)
        for i in range(self.n_attention_heads):
            attention_scores = np.dot(X, self.attention_weights_[i])
            attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=1, keepdims=True)
            attended_similarities += attention_weights[:, np.newaxis] * similarities

        attended_similarities /= self.n_attention_heads

        # Make predictions based on similarity
        y_pred = np.zeros(X.shape[0], dtype=self.classes_.dtype)
        for i in range(X.shape[0]):
            similar_indices = np.where(attended_similarities[i] > self.similarity_threshold)[0]
            if len(similar_indices) > 0:
                y_pred[i] = np.argmax(np.bincount(self.y_[similar_indices]))
            else:
                y_pred[i] = np.random.choice(self.classes_)

        return y_pred

    def score(self, X, y):
        # Predict and calculate accuracy
        y_pred = self.predict(X)
        return np.mean(y_pred == y)