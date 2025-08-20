import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

class HybridNeuron:
    def __init__(self, input_dim, threshold=0.5):
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn()
        self.threshold = threshold
        self.mode = 'continuous'  # Default mode

    def activate(self, x):
        z = np.dot(x, self.weights) + self.bias
        if self.mode == 'continuous':
            return 1 / (1 + np.exp(-z))  # Sigmoid activation
        else:  # Discrete mode
            return 1 if z > self.threshold else 0

    def switch_mode(self):
        self.mode = 'discrete' if self.mode == 'continuous' else 'continuous'

class HybridNeuronModel(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neurons=10, learning_rate=0.01, n_epochs=100, mode_switch_freq=10):
        self.n_neurons = n_neurons
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.mode_switch_freq = mode_switch_freq

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        
        # Store the number of features
        self.n_features_in_ = X.shape[1]
        
        # Initialize neurons
        self.neurons_ = [HybridNeuron(self.n_features_in_) for _ in range(self.n_neurons)]
        
        # Training loop
        for epoch in range(self.n_epochs):
            for i in range(len(X)):
                # Forward pass
                outputs = [neuron.activate(X[i]) for neuron in self.neurons_]
                prediction = np.mean(outputs)
                
                # Backward pass (simple gradient descent)
                error = y[i] - prediction
                for neuron in self.neurons_:
                    neuron.weights += self.learning_rate * error * X[i]
                    neuron.bias += self.learning_rate * error
            
            # Switch neuron modes periodically
            if epoch % self.mode_switch_freq == 0:
                for neuron in self.neurons_:
                    neuron.switch_mode()
        
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)
        
        # Input validation
        X = check_array(X)
        
        # Make predictions
        predictions = []
        for x in X:
            outputs = [neuron.activate(x) for neuron in self.neurons_]
            pred = np.mean(outputs)
            predictions.append(self.classes_[int(pred > 0.5)])
        
        return np.array(predictions)

# Example usage:
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# 
# X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 
# model = HybridNeuronModel(n_neurons=20, learning_rate=0.01, n_epochs=100, mode_switch_freq=10)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# 
# from sklearn.metrics import accuracy_score
# print(f"Accuracy: {accuracy_score(y_test, predictions)}")