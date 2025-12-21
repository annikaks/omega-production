import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class Classifier(nn.Module):
    def __init__(self, encoding_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

class CompressionDrivenLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, encoding_dim=32, num_epochs=100, batch_size=32, learning_rate=0.001):
        self.encoding_dim = encoding_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y):
        # Preprocess the data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        # Initialize models
        self.autoencoder = Autoencoder(X.shape[1], self.encoding_dim).to(self.device)
        self.classifier = Classifier(self.encoding_dim, len(np.unique(y))).to(self.device)

        # Define loss functions and optimizers
        criterion_ae = nn.MSELoss()
        criterion_clf = nn.CrossEntropyLoss()
        optimizer_ae = optim.Adam(self.autoencoder.parameters(), lr=self.learning_rate)
        optimizer_clf = optim.Adam(self.classifier.parameters(), lr=self.learning_rate)

        # Training loop
        for epoch in range(self.num_epochs):
            for i in range(0, len(X), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                batch_y = y_tensor[i:i+self.batch_size]

                # Autoencoder forward pass
                encoded, decoded = self.autoencoder(batch_X)
                loss_ae = criterion_ae(decoded, batch_X)

                # Classifier forward pass
                outputs = self.classifier(encoded.detach())
                loss_clf = criterion_clf(outputs, batch_y)

                # Backpropagation and optimization
                optimizer_ae.zero_grad()
                loss_ae.backward()
                optimizer_ae.step()

                optimizer_clf.zero_grad()
                loss_clf.backward()
                optimizer_clf.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}], AE Loss: {loss_ae.item():.4f}, CLF Loss: {loss_clf.item():.4f}")

        return self

    def predict(self, X):
        # Preprocess the data
        X_scaled = self.scaler.transform(X)
        
        # Convert data to PyTorch tensor
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Get predictions
        self.autoencoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            encoded, _ = self.autoencoder(X_tensor)
            outputs = self.classifier(encoded)
            _, predicted = torch.max(outputs.data, 1)

        return predicted.cpu().numpy()