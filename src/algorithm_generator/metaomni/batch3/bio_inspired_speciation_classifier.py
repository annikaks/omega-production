import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class Species:
    """Represents a species with a specific compression scheme and classifier."""
    
    def __init__(self, species_id: int, compression_type: str, n_components: int):
        self.species_id = species_id
        self.compression_type = compression_type
        self.n_components = n_components
        self.compressor = None
        self.classifier = None
        self.scaler = StandardScaler()
        self.fitness = 0.0
        self.niche_center = None
        self.niche_samples = []
        self.age = 0
        
    def initialize_compressor(self, n_features: int):
        """Initialize the compression scheme."""
        n_comp = min(self.n_components, n_features)
        
        if self.compression_type == 'pca':
            self.compressor = PCA(n_components=n_comp, random_state=42)
        elif self.compression_type == 'ica':
            self.compressor = FastICA(n_components=n_comp, random_state=42, max_iter=500)
        elif self.compression_type == 'nmf':
            self.compressor = NMF(n_components=n_comp, random_state=42, max_iter=500)
        elif self.compression_type == 'isomap':
            n_neighbors = min(10, n_comp + 1)
            self.compressor = Isomap(n_components=n_comp, n_neighbors=n_neighbors)
        else:
            self.compressor = PCA(n_components=n_comp, random_state=42)
            
    def initialize_classifier(self):
        """Initialize the classifier for this species."""
        self.classifier = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42 + self.species_id
        )
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the species on its niche data."""
        if len(X) == 0:
            return
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle NMF requirement for non-negative data
        if self.compression_type == 'nmf':
            X_scaled = X_scaled - X_scaled.min() + 1e-10
        
        # Compress
        try:
            X_compressed = self.compressor.fit_transform(X_scaled)
        except:
            # Fallback to PCA if compression fails
            self.compressor = PCA(n_components=self.n_components, random_state=42)
            X_compressed = self.compressor.fit_transform(X_scaled)
        
        # Train classifier
        self.classifier.fit(X_compressed, y)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the species' compression and classifier."""
        if len(X) == 0:
            return np.array([])
        
        X_scaled = self.scaler.transform(X)
        
        if self.compression_type == 'nmf':
            X_scaled = X_scaled - X_scaled.min() + 1e-10
        
        X_compressed = self.compressor.transform(X_scaled)
        return self.classifier.predict(X_compressed)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if len(X) == 0:
            return np.array([])
        
        X_scaled = self.scaler.transform(X)
        
        if self.compression_type == 'nmf':
            X_scaled = X_scaled - X_scaled.min() + 1e-10
        
        X_compressed = self.compressor.transform(X_scaled)
        return self.classifier.predict_proba(X_compressed)


class BioInspiredSpeciationClassifier(BaseEstimator, ClassifierMixin):
    """
    A bio-inspired classifier that maintains diverse compression schemes through speciation.
    Each species optimizes for different regions of the data manifold.
    """
    
    def __init__(
        self,
        n_species: int = 5,
        n_generations: int = 3,
        compatibility_threshold: float = 0.3,
        mutation_rate: float = 0.2,
        n_components_range: Tuple[int, int] = (2, 10),
        random_state: int = 42
    ):
        """
        Parameters:
        -----------
        n_species : int
            Number of species to maintain
        n_generations : int
            Number of evolutionary generations
        compatibility_threshold : float
            Threshold for assigning samples to species niches
        mutation_rate : float
            Probability of mutation during evolution
        n_components_range : tuple
            Range for number of compression components
        random_state : int
            Random seed
        """
        self.n_species = n_species
        self.n_generations = n_generations
        self.compatibility_threshold = compatibility_threshold
        self.mutation_rate = mutation_rate
        self.n_components_range = n_components_range
        self.random_state = random_state
        self.species_list: List[Species] = []
        self.classes_ = None
        
    def _initialize_species(self, n_features: int):
        """Initialize diverse species with different compression schemes."""
        np.random.seed(self.random_state)
        compression_types = ['pca', 'ica', 'nmf', 'isomap']
        
        self.species_list = []
        for i in range(self.n_species):
            compression_type = compression_types[i % len(compression_types)]
            n_components = np.random.randint(
                self.n_components_range[0],
                min(self.n_components_range[1], n_features) + 1
            )
            
            species = Species(i, compression_type, n_components)
            species.initialize_compressor(n_features)
            species.initialize_classifier()
            self.species_list.append(species)
            
    def _assign_niches(self, X: np.ndarray, y: np.ndarray):
        """Assign data samples to species niches based on manifold regions."""
        n_samples = len(X)
        
        # Use clustering to identify manifold regions
        n_clusters = min(self.n_species * 2, n_samples)
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Assign clusters to species
        clusters_per_species = n_clusters // self.n_species
        
        for i, species in enumerate(self.species_list):
            species.niche_samples = []
            start_cluster = i * clusters_per_species
            end_cluster = start_cluster + clusters_per_species
            
            if i == len(self.species_list) - 1:
                end_cluster = n_clusters
            
            for cluster_id in range(start_cluster, end_cluster):
                mask = cluster_labels == cluster_id
                species.niche_samples.extend(np.where(mask)[0])
            
            # Store niche center
            if len(species.niche_samples) > 0:
                species.niche_center = X[species.niche_samples].mean(axis=0)
            else:
                species.niche_center = X[np.random.randint(0, n_samples)]
                species.niche_samples = [np.random.randint(0, n_samples)]
                
    def _evaluate_fitness(self, X: np.ndarray, y: np.ndarray):
        """Evaluate fitness of each species on validation data."""
        for species in self.species_list:
            if len(species.niche_samples) == 0:
                species.fitness = 0.0
                continue
            
            # Use niche samples for evaluation
            niche_indices = species.niche_samples[:min(len(species.niche_samples), 100)]
            X_niche = X[niche_indices]
            y_niche = y[niche_indices]
            
            try:
                y_pred = species.predict(X_niche)
                species.fitness = accuracy_score(y_niche, y_pred)
            except:
                species.fitness = 0.0
                
    def _evolve_species(self, n_features: int):
        """Evolve species through mutation and selection."""
        # Sort by fitness
        self.species_list.sort(key=lambda s: s.fitness, reverse=True)
        
        # Keep top performers
        n_survivors = max(2, self.n_species // 2)
        survivors = self.species_list[:n_survivors]
        
        # Generate new species through mutation
        new_species_list = survivors.copy()
        
        while len(new_species_list) < self.n_species:
            # Select parent
            parent = survivors[np.random.randint(0, len(survivors))]
            
            # Create offspring with mutation
            offspring_id = len(new_species_list)
            
            if np.random.random() < self.mutation_rate:
                # Mutate compression type
                compression_types = ['pca', 'ica', 'nmf', 'isomap']
                compression_type = np.random.choice(compression_types)
            else:
                compression_type = parent.compression_type
            
            if np.random.random() < self.mutation_rate:
                # Mutate n_components
                n_components = np.random.randint(
                    self.n_components_range[0],
                    min(self.n_components_range[1], n_features) + 1
                )
            else:
                n_components = parent.n_components
            
            offspring = Species(offspring_id, compression_type, n_components)
            offspring.initialize_compressor(n_features)
            offspring.initialize_classifier()
            new_species_list.append(offspring)
        
        self.species_list = new_species_list
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fit the bio-inspired speciation classifier.
        
        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            Training data
        y_train : array-like of shape (n_samples,)
            Target values
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        self.classes_ = np.unique(y_train)
        n_features = X_train.shape[1]
        
        # Initialize species
        self._initialize_species(n_features)
        
        # Evolutionary loop
        for generation in range(self.n_generations):
            # Assign niches
            self._assign_niches(X_train, y_train)
            
            # Train each species on its niche
            for species in self.species_list:
                if len(species.niche_samples) > 0:
                    niche_indices = species.niche_samples
                    X_niche = X_train[niche_indices]
                    y_niche = y_train[niche_indices]
                    species.fit(X_niche, y_niche)
            
            # Evaluate fitness
            self._evaluate_fitness(X_train, y_train)
            
            # Evolve (except last generation)
            if generation < self.n_generations - 1:
                self._evolve_species(n_features)
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X_test.
        
        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        X_test = np.array(X_test)
        n_samples = len(X_test)
        
        # Collect predictions from all species
        all_predictions = []
        weights = []
        
        for species in self.species_list:
            if species.fitness > 0:
                try:
                    pred = species.predict(X_test)
                    all_predictions.append(pred)
                    weights.append(species.fitness)
                except:
                    pass
        
        if len(all_predictions) == 0:
            # Fallback: use best species
            best_species = max(self.species_list, key=lambda s: s.fitness)
            return best_species.predict(X_test)
        
        # Weighted voting
        all_predictions = np.array(all_predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        final_predictions = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            votes = {}
            for j, pred in enumerate(all_predictions[:, i]):
                votes[pred] = votes.get(pred, 0) + weights[j]
            final_predictions[i] = max(votes.items(), key=lambda x: x[1])[0]
        
        return final_predictions
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X_test.
        
        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        proba : array of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        X_test = np.array(X_test)
        n_samples = len(X_test)
        n_classes = len(self.classes_)
        
        # Aggregate probabilities from all species
        aggregated_proba = np.zeros((n_samples, n_classes))
        total_weight = 0.0
        
        for species in self.species_list:
            if species.fitness > 0:
                try:
                    proba = species.predict_proba(X_test)
                    aggregated_proba += proba * species.fitness
                    total_weight += species.fitness
                except:
                    pass
        
        if total_weight > 0:
            aggregated_proba /= total_weight
        else:
            # Fallback
            best_species = max(self.species_list, key=lambda s: s.fitness)
            aggregated_proba = best_species.predict_proba(X_test)
        
        return aggregated_proba