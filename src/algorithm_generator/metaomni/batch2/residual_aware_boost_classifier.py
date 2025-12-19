import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.special import expit, logit, softmax


class ResidualAwareWeakLearner:
    """
    Weak learner that models both predictions and residual error distribution.
    """
    
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.predictor = DecisionTreeRegressor(max_depth=max_depth)
        self.error_model = DecisionTreeRegressor(max_depth=max_depth)
        self.error_mean = 0.0
        self.error_std = 1.0
        
    def fit(self, X, residuals, previous_errors=None):
        """
        Fit both the predictor and error distribution model.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        residuals : array-like, shape (n_samples,)
            Current residuals to fit
        previous_errors : array-like, shape (n_samples,), optional
            Errors from previous iterations for modeling error distribution
        """
        # Fit main predictor on residuals
        self.predictor.fit(X, residuals)
        predictions = self.predictor.predict(X)
        
        # Calculate prediction errors
        errors = residuals - predictions
        
        # If we have previous errors, combine them for better error modeling
        if previous_errors is not None and len(previous_errors) > 0:
            combined_errors = np.concatenate([errors, previous_errors])
            self.error_mean = np.mean(combined_errors)
            self.error_std = np.std(combined_errors) + 1e-8
        else:
            self.error_mean = np.mean(errors)
            self.error_std = np.std(errors) + 1e-8
        
        # Normalize errors for modeling
        normalized_errors = (errors - self.error_mean) / self.error_std
        
        # Fit error distribution model (predicts expected absolute error)
        abs_errors = np.abs(normalized_errors)
        self.error_model.fit(X, abs_errors)
        
        return predictions, errors
    
    def predict(self, X):
        """Predict residuals."""
        return self.predictor.predict(X)
    
    def predict_with_uncertainty(self, X):
        """
        Predict residuals along with uncertainty estimates.
        
        Returns:
        --------
        predictions : array-like
            Predicted residuals
        uncertainties : array-like
            Estimated prediction uncertainties (standard deviations)
        """
        predictions = self.predictor.predict(X)
        # Predict expected absolute error (normalized)
        expected_abs_error = self.error_model.predict(X)
        # Convert back to original scale
        uncertainties = expected_abs_error * self.error_std
        return predictions, uncertainties


class ResidualAwareBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    Residual-Aware Boosting Classifier that explicitly models error distributions.
    
    This classifier builds an ensemble of weak learners where each learner not only
    predicts the residuals but also models the error distribution from previous
    iterations. This allows for more informed weighting and better handling of
    difficult samples.
    
    Supports both binary and multi-class classification using One-vs-Rest strategy.
    
    Parameters:
    -----------
    n_estimators : int, default=50
        Number of boosting iterations
    learning_rate : float, default=0.1
        Shrinkage parameter for each weak learner contribution
    max_depth : int, default=3
        Maximum depth of weak learner trees
    error_aware_weight : float, default=0.5
        Weight given to error-aware sample weighting (0=standard, 1=fully error-aware)
    random_state : int, default=None
        Random seed for reproducibility
    """
    
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3,
                 error_aware_weight=0.5, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.error_aware_weight = error_aware_weight
        self.random_state = random_state
        
    def _fit_binary(self, X_train, y_binary):
        """
        Fit a binary classifier.
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            Training data
        y_binary : array-like, shape (n_samples,)
            Binary target values (0 or 1)
            
        Returns:
        --------
        learners : list
            List of fitted weak learners
        learner_weights : list
            List of learner weights
        init_prediction : float
            Initial prediction value
        """
        n_samples = X_train.shape[0]
        
        # Initialize predictions with log-odds of class prior
        class_prior = np.mean(y_binary)
        class_prior = np.clip(class_prior, 1e-7, 1 - 1e-7)
        init_prediction = logit(class_prior)
        
        # Current predictions in log-odds space
        F = np.full(n_samples, init_prediction)
        
        # Store weak learners and their weights
        learners = []
        learner_weights = []
        
        # Track errors across iterations for error distribution modeling
        error_history = []
        
        for iteration in range(self.n_estimators):
            # Convert to probabilities
            probs = expit(F)
            
            # Calculate residuals (negative gradient for log loss)
            residuals = y_binary - probs
            
            # Create weak learner
            learner = ResidualAwareWeakLearner(max_depth=self.max_depth)
            
            # Fit with error history
            previous_errors = np.concatenate(error_history) if error_history else None
            predictions, errors = learner.fit(X_train, residuals, previous_errors)
            
            # Get uncertainty estimates
            _, uncertainties = learner.predict_with_uncertainty(X_train)
            
            # Calculate adaptive learning rate based on error distribution
            # Samples with higher uncertainty get more conservative updates
            uncertainty_factor = 1.0 / (1.0 + uncertainties)
            
            # Combine standard residual fitting with error-aware weighting
            standard_weight = 1.0
            error_aware_factor = uncertainty_factor
            
            combined_weight = (1 - self.error_aware_weight) * standard_weight + \
                            self.error_aware_weight * error_aware_factor
            
            # Update predictions with adaptive weighting
            update = self.learning_rate * predictions * combined_weight
            F += update
            
            # Store learner and its effective weight
            learners.append(learner)
            learner_weights.append(combined_weight)
            
            # Update error history (keep last 3 iterations to avoid memory issues)
            error_history.append(errors)
            if len(error_history) > 3:
                error_history.pop(0)
        
        return learners, learner_weights, init_prediction
    
    def fit(self, X_train, y_train):
        """
        Fit the residual-aware boosting classifier.
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            Training data
        y_train : array-like, shape (n_samples,)
            Target values
        """
        # Validate input
        X_train, y_train = check_X_y(X_train, y_train)
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        
        # Initialize random state
        np.random.seed(self.random_state)
        
        if self.n_classes_ == 2:
            # Binary classification
            y_binary = (y_train == self.classes_[1]).astype(float)
            learners, learner_weights, init_pred = self._fit_binary(X_train, y_binary)
            
            self.learners_ = [learners]
            self.learner_weights_ = [learner_weights]
            self.init_predictions_ = [init_pred]
            
        else:
            # Multi-class classification using One-vs-Rest
            self.learners_ = []
            self.learner_weights_ = []
            self.init_predictions_ = []
            
            for class_idx in range(self.n_classes_):
                # Create binary target for this class
                y_binary = (y_train == self.classes_[class_idx]).astype(float)
                
                # Fit binary classifier for this class
                learners, learner_weights, init_pred = self._fit_binary(X_train, y_binary)
                
                self.learners_.append(learners)
                self.learner_weights_.append(learner_weights)
                self.init_predictions_.append(init_pred)
        
        self.is_fitted_ = True
        return self
    
    def _predict_proba_binary(self, X_test, learners, learner_weights, init_prediction):
        """
        Predict probabilities for binary classification.
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test data
        learners : list
            List of weak learners
        learner_weights : list
            List of learner weights
        init_prediction : float
            Initial prediction value
            
        Returns:
        --------
        probs : array-like, shape (n_samples,)
            Predicted probabilities for positive class
        """
        n_samples = X_test.shape[0]
        F = np.full(n_samples, init_prediction)
        
        # Accumulate predictions from all learners
        for learner, weights in zip(learners, learner_weights):
            predictions, uncertainties = learner.predict_with_uncertainty(X_test)
            
            # Apply error-aware weighting
            uncertainty_factor = 1.0 / (1.0 + uncertainties)
            combined_weight = (1 - self.error_aware_weight) * 1.0 + \
                            self.error_aware_weight * uncertainty_factor
            
            update = self.learning_rate * predictions * combined_weight
            F += update
        
        # Convert to probabilities
        probs = expit(F)
        return probs
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        proba : array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        check_is_fitted(self, 'is_fitted_')
        X_test = check_array(X_test)
        
        n_samples = X_test.shape[0]
        
        if self.n_classes_ == 2:
            # Binary classification
            probs_pos = self._predict_proba_binary(
                X_test, 
                self.learners_[0], 
                self.learner_weights_[0], 
                self.init_predictions_[0]
            )
            return np.vstack([1 - probs_pos, probs_pos]).T
        
        else:
            # Multi-class classification using One-vs-Rest
            probs_all = np.zeros((n_samples, self.n_classes_))
            
            for class_idx in range(self.n_classes_):
                probs_all[:, class_idx] = self._predict_proba_binary(
                    X_test,
                    self.learners_[class_idx],
                    self.learner_weights_[class_idx],
                    self.init_predictions_[class_idx]
                )
            
            # Normalize probabilities using softmax
            # Convert to log-odds first for numerical stability
            log_odds = np.log(probs_all + 1e-10) - np.log(1 - probs_all + 1e-10)
            probs_normalized = softmax(log_odds, axis=1)
            
            return probs_normalized
    
    def predict(self, X_test):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X_test : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X_test)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def get_feature_importance(self):
        """
        Get feature importance scores aggregated across all learners.
        
        Returns:
        --------
        importance : array-like, shape (n_features,)
            Feature importance scores
        """
        check_is_fitted(self, 'is_fitted_')
        
        importance = None
        total_learners = 0
        
        # Aggregate importance across all classes
        for class_learners in self.learners_:
            for learner in class_learners:
                if hasattr(learner.predictor, 'feature_importances_'):
                    if importance is None:
                        importance = learner.predictor.feature_importances_.copy()
                    else:
                        importance += learner.predictor.feature_importances_
                    total_learners += 1
        
        if importance is not None and total_learners > 0:
            importance /= total_learners
        
        return importance