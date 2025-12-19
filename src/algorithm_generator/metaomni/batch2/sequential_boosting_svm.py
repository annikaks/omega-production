import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import type_of_target


class SequentialBoostingSVM(BaseEstimator, ClassifierMixin):
    """
    Sequential Boosting SVM classifier where each new SVM focuses on support vectors
    misclassified by previous models using residual-weighted kernel matrices.
    
    Parameters
    ----------
    n_estimators : int, default=5
        Number of sequential SVM models to train.
    
    C : float, default=1.0
        Regularization parameter for SVM.
    
    gamma : float or 'scale', default='scale'
        Kernel coefficient for RBF kernel.
    
    learning_rate : float, default=1.0
        Shrinks the contribution of each classifier.
    
    residual_weight_power : float, default=2.0
        Power to apply to residuals when weighting kernel matrices.
    
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, n_estimators=5, C=1.0, gamma='scale', 
                 learning_rate=1.0, residual_weight_power=2.0, random_state=None):
        self.n_estimators = n_estimators
        self.C = C
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.residual_weight_power = residual_weight_power
        self.random_state = random_state
        
    def _compute_weighted_kernel(self, X1, X2, weights=None, gamma=None):
        """Compute residual-weighted kernel matrix."""
        if gamma is None:
            gamma = self.gamma_
        
        # Compute base RBF kernel
        K = rbf_kernel(X1, X2, gamma=gamma)
        
        # Apply residual weights if provided
        if weights is not None:
            # Weight rows by residuals
            weights_sqrt = np.sqrt(weights).reshape(-1, 1)
            K = K * weights_sqrt
            
        return K
    
    def _compute_residuals(self, y_true, y_pred, sample_weights=None):
        """Compute residuals (misclassification scores) for weighting."""
        # Binary indicator of misclassification
        misclassified = (y_true != y_pred).astype(float)
        
        # Apply power transformation to emphasize misclassifications
        residuals = np.power(misclassified + 1e-10, self.residual_weight_power)
        
        # Apply existing sample weights if provided
        if sample_weights is not None:
            residuals = residuals * sample_weights
        
        # Normalize
        residuals = residuals / (np.sum(residuals) + 1e-10)
        residuals = residuals * len(residuals)  # Scale back to sum to n_samples
        
        return residuals
    
    def fit(self, X_train, y_train):
        """
        Fit the Sequential Boosting SVM model.
        
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
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y_train)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)
        
        # Determine if binary or multiclass
        self.is_binary_ = self.n_classes_ == 2
        
        # Compute gamma if 'scale'
        if self.gamma == 'scale':
            self.gamma_ = 1.0 / (X_train.shape[1] * X_train.var())
        else:
            self.gamma_ = self.gamma
        
        # Initialize storage
        self.models_ = []
        self.model_weights_ = []
        self.support_vectors_list_ = []
        self.support_vector_indices_list_ = []
        
        # Initialize sample weights (uniform)
        sample_weights = np.ones(len(X_train))
        
        # Track ensemble predictions
        ensemble_predictions = None
        
        for i in range(self.n_estimators):
            # Train SVM with current sample weights
            if self.is_binary_:
                # For binary classification, use decision_function
                model = SVC(
                    kernel='rbf',
                    C=self.C,
                    gamma=self.gamma_,
                    random_state=self.random_state,
                    probability=False
                )
            else:
                # For multiclass, use predict
                model = SVC(
                    kernel='rbf',
                    C=self.C,
                    gamma=self.gamma_,
                    random_state=self.random_state,
                    decision_function_shape='ovr',
                    probability=False
                )
            
            # Fit with sample weights
            model.fit(X_train, y_encoded, sample_weight=sample_weights)
            
            # Get predictions
            y_pred = model.predict(X_train)
            
            # Compute weighted accuracy
            correct = (y_pred == y_encoded).astype(float)
            weighted_accuracy = np.sum(correct * sample_weights) / np.sum(sample_weights)
            
            # Compute model weight (alpha) similar to AdaBoost
            epsilon = 1e-10
            weighted_error = 1 - weighted_accuracy
            weighted_error = np.clip(weighted_error, epsilon, 1 - epsilon)
            
            # For multiclass, use modified alpha calculation
            if self.is_binary_:
                alpha = self.learning_rate * 0.5 * np.log((1 - weighted_error) / weighted_error)
            else:
                # SAMME algorithm for multiclass
                alpha = self.learning_rate * (np.log((1 - weighted_error) / weighted_error) + 
                                              np.log(self.n_classes_ - 1))
            
            # Ensure alpha is positive
            alpha = max(alpha, 0)
            
            # Store model and weight
            self.models_.append(model)
            self.model_weights_.append(alpha)
            
            # Store support vectors for this model
            if hasattr(model, 'support_'):
                self.support_vector_indices_list_.append(model.support_)
                self.support_vectors_list_.append(X_train[model.support_])
            
            # Update sample weights based on residuals (focus on misclassified samples)
            if i < self.n_estimators - 1:  # Don't need to update after last iteration
                sample_weights = self._compute_residuals(y_encoded, y_pred, sample_weights)
        
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
        X_test = np.asarray(X_test)
        
        if self.is_binary_:
            # For binary classification, use decision function
            decision = self.decision_function(X_test)
            y_pred_encoded = (decision > 0).astype(int)
        else:
            # For multiclass, use weighted voting
            # Initialize vote matrix
            votes = np.zeros((len(X_test), self.n_classes_))
            
            for model, weight in zip(self.models_, self.model_weights_):
                predictions = model.predict(X_test)
                for i, pred in enumerate(predictions):
                    votes[i, pred] += weight
            
            # Get class with most votes
            y_pred_encoded = np.argmax(votes, axis=1)
        
        # Decode labels
        y_pred = self.label_encoder_.inverse_transform(y_pred_encoded)
        
        return y_pred
    
    def decision_function(self, X_test):
        """
        Compute the decision function for samples in X_test.
        Only works for binary classification.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        decision : array-like of shape (n_samples,)
            Decision function values.
        """
        X_test = np.asarray(X_test)
        
        if not self.is_binary_:
            raise ValueError("decision_function only available for binary classification")
        
        ensemble_predictions = np.zeros(len(X_test))
        
        for model, weight in zip(self.models_, self.model_weights_):
            predictions = model.decision_function(X_test)
            ensemble_predictions += weight * predictions
        
        return ensemble_predictions
    
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
            Class probabilities.
        """
        X_test = np.asarray(X_test)
        
        if self.is_binary_:
            # For binary classification, use sigmoid on decision function
            decision = self.decision_function(X_test)
            proba_positive = 1 / (1 + np.exp(-decision))
            proba_negative = 1 - proba_positive
            return np.column_stack([proba_negative, proba_positive])
        else:
            # For multiclass, use softmax on weighted votes
            votes = np.zeros((len(X_test), self.n_classes_))
            
            for model, weight in zip(self.models_, self.model_weights_):
                predictions = model.predict(X_test)
                for i, pred in enumerate(predictions):
                    votes[i, pred] += weight
            
            # Apply softmax
            exp_votes = np.exp(votes - np.max(votes, axis=1, keepdims=True))
            proba = exp_votes / np.sum(exp_votes, axis=1, keepdims=True)
            
            return proba