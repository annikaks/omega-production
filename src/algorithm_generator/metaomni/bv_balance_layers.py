import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class BiasVarianceBalancingLayer:
    def __init__(self, input_dim, output_dim, lambda_param=0.5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lambda_param = lambda_param
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.zeros((1, output_dim))

    def forward(self, X):
        return np.dot(X, self.weights) + self.bias

    def backward(self, X, grad_output, learning_rate):
        grad_weights = np.dot(X.T, grad_output)
        grad_bias = np.sum(grad_output, axis=0, keepdims=True)

        # Apply bias-variance balancing
        bias_term = self.lambda_param * self.weights
        variance_term = (1 - self.lambda_param) * grad_weights

        self.weights -= learning_rate * (bias_term + variance_term)
        self.bias -= learning_rate * grad_bias

class BiasVarianceBalancingNet(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dim=64, num_layers=3, learning_rate=0.01, epochs=100, batch_size=32, lambda_param=0.5):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_param = lambda_param

    def _init_layers(self, input_dim, output_dim):
        self.layers = []
        for i in range(self.num_layers):
            if i == 0:
                layer = BiasVarianceBalancingLayer(input_dim, self.hidden_dim, self.lambda_param)
            elif i == self.num_layers - 1:
                layer = BiasVarianceBalancingLayer(self.hidden_dim, output_dim, self.lambda_param)
            else:
                layer = BiasVarianceBalancingLayer(self.hidden_dim, self.hidden_dim, self.lambda_param)
            self.layers.append(layer)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self._init_layers(n_features, n_classes)

        # One-hot encode the target variable
        y_encoded = np.eye(n_classes)[y]

        X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        for epoch in range(self.epochs):
            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]

                # Forward pass
                activations = [X_batch]
                for layer in self.layers:
                    activations.append(layer.forward(activations[-1]))

                # Backward pass
                grad = activations[-1] - y_batch
                for i in reversed(range(len(self.layers))):
                    grad = self.layers[i].backward(activations[i], grad, self.learning_rate)

            # Compute validation loss
            val_pred = self.predict_proba(X_val)
            val_loss = mean_squared_error(y_val, val_pred)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Validation Loss: {val_loss:.4f}")

        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        activations = X
        for layer in self.layers:
            activations = layer.forward(activations)

        return activations

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]