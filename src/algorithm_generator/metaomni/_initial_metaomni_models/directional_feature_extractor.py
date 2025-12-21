import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.signal import convolve2d

class DirectionalFeatureExtractor(BaseEstimator, ClassifierMixin):
    def __init__(self, n_orientations=4, kernel_size=5, n_features=100):
        self.n_orientations = n_orientations
        self.kernel_size = kernel_size
        self.n_features = n_features

    def _create_gabor_kernels(self):
        kernels = []
        for i in range(self.n_orientations):
            theta = i * np.pi / self.n_orientations
            kernel = np.zeros((self.kernel_size, self.kernel_size))
            for x in range(self.kernel_size):
                for y in range(self.kernel_size):
                    x0 = x - self.kernel_size // 2
                    y0 = y - self.kernel_size // 2
                    x1 = x0 * np.cos(theta) + y0 * np.sin(theta)
                    y1 = -x0 * np.sin(theta) + y0 * np.cos(theta)
                    kernel[x, y] = np.exp(-(x1**2 + y1**2) / (2 * (self.kernel_size/4)**2)) * np.cos(2 * np.pi * x1 / self.kernel_size)
            kernels.append(kernel)
        return kernels

    def _extract_features(self, X):
        features = []
        for image in X:
            image_features = []
            for kernel in self.kernels:
                convolved = convolve2d(image, kernel, mode='valid')
                image_features.extend(convolved.ravel())
            features.append(image_features)
        return np.array(features)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y

        self.kernels = self._create_gabor_kernels()
        X_features = self._extract_features(X)

        # Select top n_features based on variance
        self.feature_indices_ = np.argsort(np.var(X_features, axis=0))[-self.n_features:]

        # Train a simple classifier (e.g., logistic regression) on the selected features
        from sklearn.linear_model import LogisticRegression
        self.classifier_ = LogisticRegression(random_state=42)
        self.classifier_.fit(X_features[:, self.feature_indices_], y)

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        X_features = self._extract_features(X)
        return self.classifier_.predict(X_features[:, self.feature_indices_])

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)

        X_features = self._extract_features(X)
        return self.classifier_.predict_proba(X_features[:, self.feature_indices_])