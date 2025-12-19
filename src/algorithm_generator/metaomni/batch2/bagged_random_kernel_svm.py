import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.stats import loguniform, uniform


class BaggedRandomKernelSVM(BaseEstimator, ClassifierMixin):
    """
    Bagged ensemble of SVM classifiers with random kernel parameter sampling.
    
    Creates diverse decision surfaces by combining bootstrap sampling with
    randomized kernel parameters for each base SVM classifier.
    
    Parameters
    ----------
    n_estimators : int, default=10
        Number of SVM classifiers in the ensemble.
    
    kernel : str, default='rbf'
        Kernel type for SVM ('rbf', 'poly', 'sigmoid').
    
    C_range : tuple, default=(0.1, 100)
        Range for sampling regularization parameter C (log-uniform).
    
    gamma_range : tuple, default=(0.001, 1.0)
        Range for sampling gamma parameter (log-uniform).
    
    degree_range : tuple, default=(2, 5)
        Range for sampling polynomial degree (uniform, integers).
    
    coef0_range : tuple, default=(-1.0, 1.0)
        Range for sampling coef0 parameter (uniform).
    
    bootstrap_ratio : float, default=1.0
        Ratio of samples to use in each bootstrap sample.
    
    random_state : int or None, default=None
        Random seed for reproducibility.
    
    n_jobs : int or None, default=None
        Number of parallel jobs (not implemented in this version).
    
    verbose : bool, default=False
        Enable verbose output.
    """
    
    def __init__(
        self,
        n_estimators=10,
        kernel='rbf',
        C_range=(0.1, 100),
        gamma_range=(0.001, 1.0),
        degree_range=(2, 5),
        coef0_range=(-1.0, 1.0),
        bootstrap_ratio=1.0,
        random_state=None,
        n_jobs=None,
        verbose=False
    ):
        self.n_estimators = n_estimators
        self.kernel = kernel
        self.C_range = C_range
        self.gamma_range = gamma_range
        self.degree_range = degree_range
        self.coef0_range = coef0_range
        self.bootstrap_ratio = bootstrap_ratio
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
    
    def _sample_kernel_params(self, rng):
        """Sample random kernel parameters."""
        params = {
            'C': loguniform.rvs(
                self.C_range[0], self.C_range[1], random_state=rng
            ),
            'kernel': self.kernel,
            'probability': False,
            'random_state': rng.randint(0, 10000)
        }
        
        if self.kernel == 'rbf':
            params['gamma'] = loguniform.rvs(
                self.gamma_range[0], self.gamma_range[1], random_state=rng
            )
        elif self.kernel == 'poly':
            params['gamma'] = loguniform.rvs(
                self.gamma_range[0], self.gamma_range[1], random_state=rng
            )
            params['degree'] = rng.randint(
                self.degree_range[0], self.degree_range[1] + 1
            )
            params['coef0'] = uniform.rvs(
                loc=self.coef0_range[0],
                scale=self.coef0_range[1] - self.coef0_range[0],
                random_state=rng
            )
        elif self.kernel == 'sigmoid':
            params['gamma'] = loguniform.rvs(
                self.gamma_range[0], self.gamma_range[1], random_state=rng
            )
            params['coef0'] = uniform.rvs(
                loc=self.coef0_range[0],
                scale=self.coef0_range[1] - self.coef0_range[0],
                random_state=rng
            )
        
        return params
    
    def fit(self, X_train, y_train):
        """
        Fit the bagged ensemble of SVM classifiers.
        
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
        
        # Store classes
        self.classes_ = np.unique(y_train)
        self.n_classes_ = len(self.classes_)
        
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        
        # Initialize storage for estimators and their parameters
        self.estimators_ = []
        self.estimator_params_ = []
        
        # Calculate bootstrap sample size
        n_samples = int(X_train.shape[0] * self.bootstrap_ratio)
        
        # Train ensemble
        for i in range(self.n_estimators):
            if self.verbose:
                print(f"Training estimator {i+1}/{self.n_estimators}")
            
            # Bootstrap sampling
            X_bootstrap, y_bootstrap = resample(
                X_train, y_train,
                n_samples=n_samples,
                random_state=rng.randint(0, 10000),
                replace=True
            )
            
            # Sample random kernel parameters
            params = self._sample_kernel_params(rng)
            
            if self.verbose:
                print(f"  Parameters: {params}")
            
            # Create and fit SVM
            svm = SVC(**params)
            svm.fit(X_bootstrap, y_bootstrap)
            
            # Store estimator and parameters
            self.estimators_.append(svm)
            self.estimator_params_.append(params)
        
        self.n_features_in_ = X_train.shape[1]
        
        return self
    
    def predict(self, X_test):
        """
        Predict class labels using majority voting.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        # Check if fitted
        check_is_fitted(self, ['estimators_', 'classes_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Collect predictions from all estimators
        predictions = np.array([
            estimator.predict(X_test) for estimator in self.estimators_
        ])
        
        # Majority voting
        y_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),
            axis=0,
            arr=predictions
        )
        
        return y_pred
    
    def predict_proba(self, X_test):
        """
        Predict class probabilities using voting proportions.
        
        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Test data.
        
        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        # Check if fitted
        check_is_fitted(self, ['estimators_', 'classes_'])
        
        # Validate input
        X_test = check_array(X_test)
        
        # Collect predictions from all estimators
        predictions = np.array([
            estimator.predict(X_test) for estimator in self.estimators_
        ])
        
        # Calculate voting proportions
        n_samples = X_test.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))
        
        for i in range(n_samples):
            votes = predictions[:, i]
            for class_idx, class_label in enumerate(self.classes_):
                proba[i, class_idx] = np.sum(votes == class_label) / self.n_estimators
        
        return proba
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'n_estimators': self.n_estimators,
            'kernel': self.kernel,
            'C_range': self.C_range,
            'gamma_range': self.gamma_range,
            'degree_range': self.degree_range,
            'coef0_range': self.coef0_range,
            'bootstrap_ratio': self.bootstrap_ratio,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self