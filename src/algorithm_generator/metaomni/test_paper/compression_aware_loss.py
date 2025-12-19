import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import zlib
import pickle
from typing import Optional, Callable
import warnings


class CompressionAwareLoss(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_estimator=None,
        compression_weight: float = 0.1,
        hidden_layer_sizes: tuple = (100,),
        activation: str = 'relu',
        solver: str = 'adam',
        alpha: float = 0.0001,
        batch_size: str = 'auto',
        learning_rate: str = 'constant',
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        random_state: Optional[int] = None,
        tol: float = 1e-4,
        verbose: bool = False,
        warm_start: bool = False,
        momentum: float = 0.9,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        compression_method: str = 'zlib',
        description_length_type: str = 'weights',
        adaptive_compression_weight: bool = False
    ):
        self.base_estimator = base_estimator
        self.compression_weight = compression_weight
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.compression_method = compression_method
        self.description_length_type = description_length_type
        self.adaptive_compression_weight = adaptive_compression_weight
        
    def _compute_description_length(self, data: np.ndarray) -> float:
        if self.compression_method == 'zlib':
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = zlib.compress(serialized, level=9)
            return len(compressed)
        elif self.compression_method == 'entropy':
            flat_data = data.flatten()
            hist, _ = np.histogram(flat_data, bins=256, density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            return entropy * len(flat_data) / 8
        else:
            raise ValueError(f"Unknown compression method: {self.compression_method}")
    
    def _compute_model_complexity(self) -> float:
        if self.description_length_type == 'weights':
            total_dl = 0
            if hasattr(self.estimator_, 'coefs_'):
                for coef in self.estimator_.coefs_:
                    total_dl += self._compute_description_length(coef)
                for intercept in self.estimator_.intercepts_:
                    total_dl += self._compute_description_length(intercept)
            else:
                params = self._get_estimator_params()
                if params is not None:
                    total_dl = self._compute_description_length(params)
            return total_dl
        elif self.description_length_type == 'activations':
            if hasattr(self, 'cached_activations_'):
                return self._compute_description_length(self.cached_activations_)
            return 0.0
        elif self.description_length_type == 'combined':
            weight_dl = 0
            if hasattr(self.estimator_, 'coefs_'):
                for coef in self.estimator_.coefs_:
                    weight_dl += self._compute_description_length(coef)
            activation_dl = 0
            if hasattr(self, 'cached_activations_'):
                activation_dl = self._compute_description_length(self.cached_activations_)
            return weight_dl + activation_dl
        else:
            raise ValueError(f"Unknown description_length_type: {self.description_length_type}")
    
    def _get_estimator_params(self) -> Optional[np.ndarray]:
        if hasattr(self.estimator_, 'coef_'):
            return self.estimator_.coef_
        elif hasattr(self.estimator_, 'feature_importances_'):
            return self.estimator_.feature_importances_
        return None
    
    def _extract_activations(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator_, '_forward_pass'):
            activations = []
            layer_units = [X.shape[1]] + list(self.estimator_.hidden_layer_sizes) + [self.estimator_.n_outputs_]
            
            if hasattr(self.estimator_, 'coefs_'):
                activation = X
                for i, (coef, intercept) in enumerate(zip(self.estimator_.coefs_, self.estimator_.intercepts_)):
                    activation = np.dot(activation, coef) + intercept
                    if i < len(self.estimator_.coefs_) - 1:
                        if self.activation == 'relu':
                            activation = np.maximum(0, activation)
                        elif self.activation == 'tanh':
                            activation = np.tanh(activation)
                        elif self.activation == 'logistic':
                            activation = 1 / (1 + np.exp(-activation))
                    activations.append(activation)
                return np.concatenate([a.flatten() for a in activations])
        return np.array([])
    
    def _custom_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        
        n_samples = len(y_true)
        n_classes = y_pred_proba.shape[1]
        
        y_true_one_hot = np.zeros((n_samples, n_classes))
        y_true_one_hot[np.arange(n_samples), y_true] = 1
        
        cross_entropy = -np.sum(y_true_one_hot * np.log(y_pred_proba)) / n_samples
        
        complexity = self._compute_model_complexity()
        
        if self.adaptive_compression_weight:
            data_size = np.prod(self.X_train_.shape)
            adaptive_weight = self.compression_weight * (complexity / (data_size + 1e-10))
        else:
            adaptive_weight = self.compression_weight
        
        total_loss = cross_entropy + adaptive_weight * complexity
        
        return total_loss
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.X_ = X
        self.y_ = y
        self.X_train_ = X
        
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        if self.base_estimator is not None:
            self.estimator_ = self.base_estimator
        else:
            self.estimator_ = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
                tol=self.tol,
                verbose=self.verbose,
                warm_start=self.warm_start,
                momentum=self.momentum,
                early_stopping=self.early_stopping,
                validation_fraction=self.validation_fraction
            )
        
        self.estimator_.fit(X_scaled, y)
        
        if self.description_length_type in ['activations', 'combined']:
            self.cached_activations_ = self._extract_activations(X_scaled[:min(1000, len(X_scaled))])
        
        self.complexity_ = self._compute_model_complexity()
        
        y_pred_proba = self.estimator_.predict_proba(X_scaled)
        self.training_loss_ = self._custom_loss(y, y_pred_proba)
        
        if self.verbose:
            print(f"Training Loss: {self.training_loss_:.6f}")
            print(f"Model Complexity (Description Length): {self.complexity_:.2f} bytes")
        
        return self
    
    def predict(self, X):
        check_is_fitted(self, ['estimator_', 'scaler_'])
        X = check_array(X)
        X_scaled = self.scaler_.transform(X)
        return self.estimator_.predict(X_scaled)
    
    def predict_proba(self, X):
        check_is_fitted(self, ['estimator_', 'scaler_'])
        X = check_array(X)
        X_scaled = self.scaler_.transform(X)
        return self.estimator_.predict_proba(X_scaled)
    
    def score(self, X, y):
        check_is_fitted(self, ['estimator_', 'scaler_'])
        X, y = check_X_y(X, y)
        X_scaled = self.scaler_.transform(X)
        return self.estimator_.score(X_scaled, y)
    
    def get_compression_metrics(self):
        check_is_fitted(self, ['estimator_'])
        
        metrics = {
            'description_length': self.complexity_,
            'training_loss': self.training_loss_,
            'compression_weight': self.compression_weight,
            'n_parameters': self._count_parameters()
        }
        
        if hasattr(self.estimator_, 'loss_'):
            metrics['base_loss'] = self.estimator_.loss_
        
        return metrics
    
    def _count_parameters(self) -> int:
        if hasattr(self.estimator_, 'coefs_'):
            total = 0
            for coef in self.estimator_.coefs_:
                total += coef.size
            for intercept in self.estimator_.intercepts_:
                total += intercept.size
            return total
        return 0
    
    def get_params(self, deep=True):
        params = {
            'base_estimator': self.base_estimator,
            'compression_weight': self.compression_weight,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'solver': self.solver,
            'alpha': self.alpha,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'learning_rate_init': self.learning_rate_init,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'tol': self.tol,
            'verbose': self.verbose,
            'warm_start': self.warm_start,
            'momentum': self.momentum,
            'early_stopping': self.early_stopping,
            'validation_fraction': self.validation_fraction,
            'compression_method': self.compression_method,
            'description_length_type': self.description_length_type,
            'adaptive_compression_weight': self.adaptive_compression_weight
        }
        return params
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("=" * 80)
    print("Compression-Aware Loss Function Classifier")
    print("=" * 80)
    
    clf = CompressionAwareLoss(
        compression_weight=0.001,
        hidden_layer_sizes=(50, 30),
        max_iter=100,
        random_state=42,
        verbose=True,
        compression_method='zlib',
        description_length_type='weights'
    )
    
    print("\nTraining model...")
    clf.fit(X_train, y_train)
    
    print("\nMaking predictions...")
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    print("\nCompression Metrics:")
    metrics = clf.get_compression_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n" + "=" * 80)
    print("Testing with different compression weights")
    print("=" * 80)
    
    for weight in [0.0, 0.0001, 0.001, 0.01]:
        clf_test = CompressionAwareLoss(
            compression_weight=weight,
            hidden_layer_sizes=(50, 30),
            max_iter=100,
            random_state=42,
            verbose=False
        )
        clf_test.fit(X_train, y_train)
        y_pred_test = clf_test.predict(X_test)
        acc = accuracy_score(y_test, y_pred_test)
        metrics_test = clf_test.get_compression_metrics()
        print(f"\nWeight: {weight:.4f} | Accuracy: {acc:.4f} | "
              f"DL: {metrics_test['description_length']:.2f} | "
              f"Params: {metrics_test['n_parameters']}")