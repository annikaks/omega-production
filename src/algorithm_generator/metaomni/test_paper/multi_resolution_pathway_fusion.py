import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from skimage.feature import hog
from skimage.filters import sobel
import warnings
warnings.filterwarnings('ignore')


class MultiResolutionPathwayFusion(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_pathways=4,
        pathway_dims=[128, 64, 32, 16],
        use_pca=True,
        pca_variance=0.95,
        base_classifier='rf',
        n_estimators=100,
        fusion_method='weighted',
        random_state=42
    ):
        self.n_pathways = n_pathways
        self.pathway_dims = pathway_dims
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.fusion_method = fusion_method
        self.random_state = random_state
        
    def _extract_fine_grained_features(self, X):
        features = []
        for sample in X:
            if len(sample.shape) == 1:
                img_size = int(np.sqrt(len(sample)))
                if img_size * img_size == len(sample):
                    img = sample.reshape(img_size, img_size)
                else:
                    features.append(sample)
                    continue
            else:
                img = sample
            
            pixel_features = img.flatten()
            features.append(pixel_features)
        
        return np.array(features)
    
    def _extract_edge_features(self, X):
        features = []
        for sample in X:
            if len(sample.shape) == 1:
                img_size = int(np.sqrt(len(sample)))
                if img_size * img_size == len(sample):
                    img = sample.reshape(img_size, img_size)
                else:
                    features.append(sample)
                    continue
            else:
                img = sample
            
            edges = sobel(img)
            edge_features = edges.flatten()
            features.append(edge_features)
        
        return np.array(features)
    
    def _extract_texture_features(self, X):
        features = []
        for sample in X:
            if len(sample.shape) == 1:
                img_size = int(np.sqrt(len(sample)))
                if img_size * img_size == len(sample):
                    img = sample.reshape(img_size, img_size)
                else:
                    img_size = 8
                    img = sample[:img_size*img_size].reshape(img_size, img_size)
            else:
                img = sample
            
            try:
                if img.shape[0] >= 8 and img.shape[1] >= 8:
                    hog_features = hog(
                        img,
                        orientations=8,
                        pixels_per_cell=(max(2, img.shape[0]//4), max(2, img.shape[1]//4)),
                        cells_per_block=(1, 1),
                        feature_vector=True
                    )
                else:
                    hog_features = img.flatten()
            except:
                hog_features = img.flatten()
            
            features.append(hog_features)
        
        return np.array(features)
    
    def _extract_coarse_features(self, X):
        features = []
        for sample in X:
            if len(sample.shape) == 1:
                img_size = int(np.sqrt(len(sample)))
                if img_size * img_size == len(sample):
                    img = sample.reshape(img_size, img_size)
                else:
                    features.append(sample[:64] if len(sample) >= 64 else np.pad(sample, (0, max(0, 64-len(sample)))))
                    continue
            else:
                img = sample
            
            target_size = max(4, min(8, img.shape[0]//4))
            try:
                coarse_img = resize(img, (target_size, target_size), anti_aliasing=True)
            except:
                coarse_img = img
            
            smoothed = gaussian_filter(coarse_img, sigma=1.0)
            coarse_features = smoothed.flatten()
            features.append(coarse_features)
        
        return np.array(features)
    
    def _create_pathway_classifier(self):
        if self.base_classifier == 'rf':
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_depth=10,
                n_jobs=-1
            )
        elif self.base_classifier == 'lr':
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=-1
            )
        elif self.base_classifier == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(100, 50),
                random_state=self.random_state,
                max_iter=500
            )
        else:
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
    
    def fit(self, X_train, y_train):
        X_train, y_train = check_X_y(X_train, y_train, accept_sparse=False)
        self.classes_ = unique_labels(y_train)
        self.n_classes_ = len(self.classes_)
        
        self.pathways_ = []
        self.scalers_ = []
        self.pca_models_ = []
        self.classifiers_ = []
        self.pathway_weights_ = []
        
        feature_extractors = [
            self._extract_fine_grained_features,
            self._extract_edge_features,
            self._extract_texture_features,
            self._extract_coarse_features
        ]
        
        for i in range(min(self.n_pathways, len(feature_extractors))):
            extractor = feature_extractors[i]
            features = extractor(X_train)
            
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            self.scalers_.append(scaler)
            
            if self.use_pca and features_scaled.shape[1] > self.pathway_dims[i]:
                pca = PCA(
                    n_components=min(self.pathway_dims[i], features_scaled.shape[1]),
                    random_state=self.random_state
                )
                features_reduced = pca.fit_transform(features_scaled)
                self.pca_models_.append(pca)
            else:
                features_reduced = features_scaled
                self.pca_models_.append(None)
            
            classifier = self._create_pathway_classifier()
            classifier.fit(features_reduced, y_train)
            self.classifiers_.append(classifier)
            
            train_score = classifier.score(features_reduced, y_train)
            self.pathway_weights_.append(train_score)
        
        total_weight = sum(self.pathway_weights_)
        if total_weight > 0:
            self.pathway_weights_ = [w / total_weight for w in self.pathway_weights_]
        else:
            self.pathway_weights_ = [1.0 / len(self.pathway_weights_)] * len(self.pathway_weights_)
        
        self.meta_classifier_ = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        
        meta_features = []
        for i in range(len(self.classifiers_)):
            extractor = feature_extractors[i]
            features = extractor(X_train)
            features_scaled = self.scalers_[i].transform(features)
            
            if self.pca_models_[i] is not None:
                features_reduced = self.pca_models_[i].transform(features_scaled)
            else:
                features_reduced = features_scaled
            
            proba = self.classifiers_[i].predict_proba(features_reduced)
            meta_features.append(proba)
        
        meta_features = np.hstack(meta_features)
        self.meta_classifier_.fit(meta_features, y_train)
        
        return self
    
    def predict(self, X_test):
        check_is_fitted(self, ['classifiers_', 'scalers_'])
        X_test = check_array(X_test, accept_sparse=False)
        
        feature_extractors = [
            self._extract_fine_grained_features,
            self._extract_edge_features,
            self._extract_texture_features,
            self._extract_coarse_features
        ]
        
        if self.fusion_method == 'weighted':
            predictions = np.zeros((X_test.shape[0], self.n_classes_))
            
            for i in range(len(self.classifiers_)):
                extractor = feature_extractors[i]
                features = extractor(X_test)
                features_scaled = self.scalers_[i].transform(features)
                
                if self.pca_models_[i] is not None:
                    features_reduced = self.pca_models_[i].transform(features_scaled)
                else:
                    features_reduced = features_scaled
                
                proba = self.classifiers_[i].predict_proba(features_reduced)
                predictions += self.pathway_weights_[i] * proba
            
            return self.classes_[np.argmax(predictions, axis=1)]
        
        elif self.fusion_method == 'voting':
            all_predictions = []
            
            for i in range(len(self.classifiers_)):
                extractor = feature_extractors[i]
                features = extractor(X_test)
                features_scaled = self.scalers_[i].transform(features)
                
                if self.pca_models_[i] is not None:
                    features_reduced = self.pca_models_[i].transform(features_scaled)
                else:
                    features_reduced = features_scaled
                
                pred = self.classifiers_[i].predict(features_reduced)
                all_predictions.append(pred)
            
            all_predictions = np.array(all_predictions).T
            final_predictions = []
            for preds in all_predictions:
                unique, counts = np.unique(preds, return_counts=True)
                final_predictions.append(unique[np.argmax(counts)])
            
            return np.array(final_predictions)
        
        else:
            meta_features = []
            for i in range(len(self.classifiers_)):
                extractor = feature_extractors[i]
                features = extractor(X_test)
                features_scaled = self.scalers_[i].transform(features)
                
                if self.pca_models_[i] is not None:
                    features_reduced = self.pca_models_[i].transform(features_scaled)
                else:
                    features_reduced = features_scaled
                
                proba = self.classifiers_[i].predict_proba(features_reduced)
                meta_features.append(proba)
            
            meta_features = np.hstack(meta_features)
            return self.meta_classifier_.predict(meta_features)
    
    def predict_proba(self, X_test):
        check_is_fitted(self, ['classifiers_', 'scalers_'])
        X_test = check_array(X_test, accept_sparse=False)
        
        feature_extractors = [
            self._extract_fine_grained_features,
            self._extract_edge_features,
            self._extract_texture_features,
            self._extract_coarse_features
        ]
        
        if self.fusion_method == 'weighted':
            predictions = np.zeros((X_test.shape[0], self.n_classes_))
            
            for i in range(len(self.classifiers_)):
                extractor = feature_extractors[i]
                features = extractor(X_test)
                features_scaled = self.scalers_[i].transform(features)
                
                if self.pca_models_[i] is not None:
                    features_reduced = self.pca_models_[i].transform(features_scaled)
                else:
                    features_reduced = features_scaled
                
                proba = self.classifiers_[i].predict_proba(features_reduced)
                predictions += self.pathway_weights_[i] * proba
            
            return predictions
        
        else:
            meta_features = []
            for i in range(len(self.classifiers_)):
                extractor = feature_extractors[i]
                features = extractor(X_test)
                features_scaled = self.scalers_[i].transform(features)
                
                if self.pca_models_[i] is not None:
                    features_reduced = self.pca_models_[i].transform(features_scaled)
                else:
                    features_reduced = features_scaled
                
                proba = self.classifiers_[i].predict_proba(features_reduced)
                meta_features.append(proba)
            
            meta_features = np.hstack(meta_features)
            return self.meta_classifier_.predict_proba(meta_features)


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    digits = load_digits()
    X, y = digits.data, digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    clf = MultiResolutionPathwayFusion(
        n_pathways=4,
        pathway_dims=[128, 64, 32, 16],
        use_pca=True,
        base_classifier='rf',
        n_estimators=50,
        fusion_method='weighted',
        random_state=42
    )
    
    print("Training Multi-Resolution Pathway Fusion Classifier...")
    clf.fit(X_train, y_train)
    
    print("\nMaking predictions...")
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    y_proba = clf.predict_proba(X_test)
    print(f"\nPrediction probabilities shape: {y_proba.shape}")
    print(f"Sample probabilities for first test instance:\n{y_proba[0]}")