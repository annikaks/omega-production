import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class AdaptiveLocalPruner(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 input_dim=None,
                 hidden_dims=[128, 64, 32],
                 output_dim=None,
                 learning_rate=0.01,
                 epochs=100,
                 batch_size=32,
                 prune_threshold=0.01,
                 prune_frequency=10,
                 region_split_depth=3,
                 bias_variance_alpha=0.5,
                 min_samples_per_region=20,
                 random_state=42):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.prune_threshold = prune_threshold
        self.prune_frequency = prune_frequency
        self.region_split_depth = region_split_depth
        self.bias_variance_alpha = bias_variance_alpha
        self.min_samples_per_region = min_samples_per_region
        self.random_state = random_state
        
        self.weights = []
        self.biases = []
        self.neuron_masks = []
        self.region_tree = None
        self.region_neuron_importance = defaultdict(lambda: defaultdict(list))
        self.classes_ = None
        
    def _initialize_network(self, input_dim, output_dim):
        np.random.seed(self.random_state)
        self.weights = []
        self.biases = []
        self.neuron_masks = []
        
        dims = [input_dim] + self.hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)
            
            if i < len(dims) - 2:
                self.neuron_masks.append(np.ones(dims[i+1], dtype=bool))
            else:
                self.neuron_masks.append(np.ones(dims[i+1], dtype=bool))
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _forward(self, X):
        activations = [X]
        pre_activations = []
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            pre_activations.append(z)
            
            if i < len(self.weights) - 1:
                a = self._relu(z)
                a[:, ~self.neuron_masks[i]] = 0
            else:
                a = self._softmax(z)
            
            activations.append(a)
        
        return activations, pre_activations
    
    def _backward(self, X, y, activations, pre_activations):
        m = X.shape[0]
        gradients_w = []
        gradients_b = []
        
        y_one_hot = np.zeros((m, self.output_dim))
        y_one_hot[np.arange(m), y] = 1
        
        delta = activations[-1] - y_one_hot
        
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w = np.dot(activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0) / m
            
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                delta[:, ~self.neuron_masks[i-1]] = 0
                delta = delta * self._relu_derivative(pre_activations[i-1])
        
        return gradients_w, gradients_b
    
    def _build_region_tree(self, X, y):
        self.region_tree = DecisionTreeClassifier(
            max_depth=self.region_split_depth,
            min_samples_leaf=self.min_samples_per_region,
            random_state=self.random_state
        )
        self.region_tree.fit(X, y)
    
    def _get_region_id(self, X):
        return self.region_tree.apply(X)
    
    def _compute_neuron_importance(self, X, y, layer_idx):
        activations, _ = self._forward(X)
        layer_activations = activations[layer_idx + 1]
        
        importance = np.abs(layer_activations).mean(axis=0)
        
        return importance
    
    def _compute_local_bias_variance(self, X, y, region_id, layer_idx, neuron_idx):
        region_mask = self._get_region_id(X) == region_id
        
        if np.sum(region_mask) < 5:
            return 0.0
        
        X_region = X[region_mask]
        y_region = y[region_mask]
        
        original_mask = self.neuron_masks[layer_idx][neuron_idx]
        
        self.neuron_masks[layer_idx][neuron_idx] = True
        activations_with, _ = self._forward(X_region)
        pred_with = np.argmax(activations_with[-1], axis=1)
        
        self.neuron_masks[layer_idx][neuron_idx] = False
        activations_without, _ = self._forward(X_region)
        pred_without = np.argmax(activations_without[-1], axis=1)
        
        self.neuron_masks[layer_idx][neuron_idx] = original_mask
        
        acc_with = np.mean(pred_with == y_region)
        acc_without = np.mean(pred_without == y_region)
        
        variance_with = np.var(activations_with[-1])
        variance_without = np.var(activations_without[-1])
        
        bias_change = acc_without - acc_with
        variance_change = variance_without - variance_with
        
        tradeoff = self.bias_variance_alpha * bias_change + (1 - self.bias_variance_alpha) * variance_change
        
        return tradeoff
    
    def _prune_neurons(self, X, y):
        region_ids = np.unique(self._get_region_id(X))
        
        for layer_idx in range(len(self.neuron_masks) - 1):
            active_neurons = np.where(self.neuron_masks[layer_idx])[0]
            
            if len(active_neurons) <= 2:
                continue
            
            importance_scores = self._compute_neuron_importance(X, y, layer_idx)
            
            for region_id in region_ids:
                region_mask = self._get_region_id(X) == region_id
                
                if np.sum(region_mask) < self.min_samples_per_region:
                    continue
                
                for neuron_idx in active_neurons:
                    if importance_scores[neuron_idx] < self.prune_threshold:
                        tradeoff = self._compute_local_bias_variance(
                            X, y, region_id, layer_idx, neuron_idx
                        )
                        
                        if tradeoff > 0:
                            self.neuron_masks[layer_idx][neuron_idx] = False
                            break
    
    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.classes_ = np.unique(y_train)
        self.input_dim = X_train.shape[1]
        self.output_dim = len(self.classes_)
        
        label_map = {label: idx for idx, label in enumerate(self.classes_)}
        y_mapped = np.array([label_map[label] for label in y_train])
        
        self._initialize_network(self.input_dim, self.output_dim)
        
        self._build_region_tree(X_train, y_mapped)
        
        n_samples = X_train.shape[0]
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_mapped[indices]
            
            for i in range(0, n_samples, self.batch_size):
                batch_X = X_shuffled[i:i+self.batch_size]
                batch_y = y_shuffled[i:i+self.batch_size]
                
                activations, pre_activations = self._forward(batch_X)
                gradients_w, gradients_b = self._backward(batch_X, batch_y, activations, pre_activations)
                
                for j in range(len(self.weights)):
                    self.weights[j] -= self.learning_rate * gradients_w[j]
                    self.biases[j] -= self.learning_rate * gradients_b[j]
            
            if (epoch + 1) % self.prune_frequency == 0:
                self._prune_neurons(X_train, y_mapped)
        
        return self
    
    def predict(self, X_test):
        X_test = np.array(X_test)
        activations, _ = self._forward(X_test)
        predictions = np.argmax(activations[-1], axis=1)
        return self.classes_[predictions]
    
    def predict_proba(self, X_test):
        X_test = np.array(X_test)
        activations, _ = self._forward(X_test)
        return activations[-1]
    
    def get_active_neurons_per_layer(self):
        return [np.sum(mask) for mask in self.neuron_masks[:-1]]
    
    def get_pruning_statistics(self):
        stats = {}
        for i, mask in enumerate(self.neuron_masks[:-1]):
            total = len(mask)
            active = np.sum(mask)
            pruned = total - active
            stats[f'layer_{i}'] = {
                'total': total,
                'active': active,
                'pruned': pruned,
                'pruning_rate': pruned / total if total > 0 else 0
            }
        return stats