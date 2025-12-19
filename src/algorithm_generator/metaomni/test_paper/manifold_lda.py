import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.manifold import Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class ManifoldLDA(BaseEstimator, ClassifierMixin):
    """
    Manifold Learning + Linear Discriminant Analysis Classifier.
    
    This classifier first applies manifold learning (Isomap) to project data
    into an intrinsic low-dimensional space, reducing noise from irrelevant
    dimensions, before applying Linear Discriminant Analysis for classification.
    
    Parameters
    ----------
    n_manifold_components : int, default=10
        Number of components for manifold learning projection.
    n_neighbors : int, default=5
        Number of neighbors for Isomap manifold learning.
    manifold_method : str, default='isomap'
        Manifold learning method to use. Currently supports 'isomap'.
    lda_solver : str, default='svd'
        Solver for LDA. Options: 'svd', 'lsqr', 'eigen'.
    lda_shrinkage : str or float, default=None
        Shrinkage parameter for LDA (only used with 'lsqr' and 'eigen' solvers).
    scale_data : bool, default=True
        Whether to standardize features before manifold learning.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    scaler_ : StandardScaler
        Fitted scaler for data standardization.
    manifold_ : Isomap
        Fitted manifold learning transformer.
    lda_ : LinearDiscriminantAnalysis
        Fitted LDA classifier.
    """
    
    def __init__(
        self,
        n_manifold_components=10,
        n_neighbors=5,
        manifold_method='isomap',
        lda_solver='svd',
        lda_shrinkage=None,
        scale_data=True
    ):
        self.n_manifold_components = n_manifold_components
        self.n_neighbors = n_neighbors
        self.manifold_method = manifold_method
        self.lda_solver = lda_solver
        self.lda_shrinkage = lda_shrinkage
        self.scale_data = scale_data
    
    def fit(self, X_train, y_train):
        """
        Fit the ManifoldLDA classifier.
        
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
        
        # Store number of features
        self.n_features_in_ = X_train.shape[1]
        
        # Step 1: Standardize data if requested
        if self.scale_data:
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_train)
        else:
            self.scaler_ = None
            X_scaled = X_train
        
        # Step 2: Apply manifold learning to reduce to intrinsic dimensions
        if self.manifold_method == 'isomap':
            self.manifold_ = Isomap(
                n_components=self.n_manifold_components,
                n_neighbors=self.n_neighbors
            )
        else:
            raise ValueError(f"Unsupported manifold method: {self.manifold_method}")
        
        X_manifold = self.manifold_.fit_transform(X_scaled)
        
        # Step 3: Apply LDA on the manifold-projected data
        self.lda_ = LinearDiscriminantAnalysis(
            solver=self.lda_solver,
            shrinkage=self.lda_shrinkage
        )
        self.lda_.fit(X_manifold, y_train)
        
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
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fitted
        check_is_fitted(self, ['manifold_', 'lda_', 'classes_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Apply the same transformations as in fit
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X_test)
        else:
            X_scaled = X_test
        
        # Transform to manifold space
        X_manifold = self.manifold_.transform(X_scaled)
        
        # Predict using LDA
        y_pred = self.lda_.predict(X_manifold)
        
        return y_pred
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        # Check if fitted
        check_is_fitted(self, ['manifold_', 'lda_', 'classes_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Apply the same transformations as in fit
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X_test)
        else:
            X_scaled = X_test
        
        # Transform to manifold space
        X_manifold = self.manifold_.transform(X_scaled)
        
        # Predict probabilities using LDA
        proba = self.lda_.predict_proba(X_manifold)
        
        return proba
    
    def decision_function(self, X_test):
        """
        Compute decision function for samples in X_test.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        decision : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function values.
        """
        # Check if fitted
        check_is_fitted(self, ['manifold_', 'lda_', 'classes_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Apply the same transformations as in fit
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X_test)
        else:
            X_scaled = X_test
        
        # Transform to manifold space
        X_manifold = self.manifold_.transform(X_scaled)
        
        # Compute decision function using LDA
        decision = self.lda_.decision_function(X_manifold)
        
        return decision