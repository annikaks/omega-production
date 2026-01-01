from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np


class EnsembleDimReduceClassifier(BaseEstimator, ClassifierMixin):
    """
    An ensemble classifier that combines Logistic Regression, HistGradientBoosting,
    and Random Forest with feature dimensionality reduction using PCA.
    """
    
    def __init__(
        self,
        n_components=None,
        variance_threshold=0.95,
        scale_features=True,
        lr_C=1.0,
        lr_max_iter=1000,
        hgb_max_iter=100,
        hgb_learning_rate=0.1,
        hgb_max_depth=None,
        rf_n_estimators=100,
        rf_max_depth=None,
        voting='soft',
        random_state=None
    ):
        """
        Initialize the ensemble classifier with dimensionality reduction.
        
        Parameters
        ----------
        n_components : int or None, default=None
            Number of components for PCA. If None, uses variance_threshold.
        variance_threshold : float, default=0.95
            Variance threshold for PCA when n_components is None.
        scale_features : bool, default=True
            Whether to scale features before PCA.
        lr_C : float, default=1.0
            Inverse of regularization strength for Logistic Regression.
        lr_max_iter : int, default=1000
            Maximum iterations for Logistic Regression.
        hgb_max_iter : int, default=100
            Maximum iterations for HistGradientBoosting.
        hgb_learning_rate : float, default=0.1
            Learning rate for HistGradientBoosting.
        hgb_max_depth : int or None, default=None
            Maximum depth for HistGradientBoosting.
        rf_n_estimators : int, default=100
            Number of trees in Random Forest.
        rf_max_depth : int or None, default=None
            Maximum depth for Random Forest.
        voting : str, default='soft'
            Voting strategy ('hard' or 'soft').
        random_state : int or None, default=None
            Random state for reproducibility.
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.scale_features = scale_features
        self.lr_C = lr_C
        self.lr_max_iter = lr_max_iter
        self.hgb_max_iter = hgb_max_iter
        self.hgb_learning_rate = hgb_learning_rate
        self.hgb_max_depth = hgb_max_depth
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.voting = voting
        self.random_state = random_state
        
    def fit(self, X_train, y_train):
        """
        Fit the ensemble classifier with dimensionality reduction.
        
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
        # Build preprocessing pipeline
        preprocessing_steps = []
        
        if self.scale_features:
            preprocessing_steps.append(('scaler', StandardScaler()))
        
        # Determine n_components for PCA
        if self.n_components is None:
            # Fit PCA to determine components based on variance
            temp_pca = PCA(n_components=self.variance_threshold, random_state=self.random_state)
            if self.scale_features:
                temp_scaler = StandardScaler()
                X_scaled = temp_scaler.fit_transform(X_train)
                temp_pca.fit(X_scaled)
            else:
                temp_pca.fit(X_train)
            n_comp = temp_pca.n_components_
        else:
            n_comp = min(self.n_components, X_train.shape[1])
        
        preprocessing_steps.append(('pca', PCA(n_components=n_comp, random_state=self.random_state)))
        
        self.preprocessor_ = Pipeline(preprocessing_steps)
        
        # Transform training data
        X_transformed = self.preprocessor_.fit_transform(X_train)
        
        # Create base estimators
        lr = LogisticRegression(
            C=self.lr_C,
            max_iter=self.lr_max_iter,
            random_state=self.random_state
        )
        
        hgb = HistGradientBoostingClassifier(
            max_iter=self.hgb_max_iter,
            learning_rate=self.hgb_learning_rate,
            max_depth=self.hgb_max_depth,
            random_state=self.random_state
        )
        
        rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            random_state=self.random_state
        )
        
        # Create voting classifier
        self.ensemble_ = VotingClassifier(
            estimators=[
                ('logistic_regression', lr),
                ('hist_gradient_boosting', hgb),
                ('random_forest', rf)
            ],
            voting=self.voting
        )
        
        # Fit ensemble on transformed data
        self.ensemble_.fit(X_transformed, y_train)
        
        # Store classes
        self.classes_ = self.ensemble_.classes_
        
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
        # Transform test data
        X_transformed = self.preprocessor_.transform(X_test)
        
        # Predict using ensemble
        return self.ensemble_.predict(X_transformed)
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
            
        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        if self.voting != 'soft':
            raise AttributeError("predict_proba is only available when voting='soft'")
        
        # Transform test data
        X_transformed = self.preprocessor_.transform(X_test)
        
        # Predict probabilities using ensemble
        return self.ensemble_.predict_proba(X_transformed)