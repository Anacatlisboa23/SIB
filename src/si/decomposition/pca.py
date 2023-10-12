import numpy as np


class PCA:
    def __init__(self, n_components):
        """
        Initialize the PCA class.

        Args:
            n_components (int): Number of principal components to retain.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        """
        Fit the PCA model to the data.

        Args:
            X (numpy array): Input data with shape (n_samples, n_features).
        """

        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices[:self.n_components]]

    def transform(self, X):
        """
        Transform the input data into the new feature space.

        Args:
            X (numpy array): Input data with shape (n_samples, n_features).

        Returns:
            numpy array: Transformed data with shape (n_samples, n_components).
        """
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
