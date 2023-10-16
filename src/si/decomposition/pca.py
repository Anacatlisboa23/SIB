import numpy as np
from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from numpy.linalg import svd


class PCA:
    def __init__(self, n_components: int):
        """
      Initialize a Principal Component Analysis (PCA) instance with a specified number of components.

      Parameters:
      n_components (int): The number of principal components to retain.

      Attributes:
      n_components (int): The number of principal components specified during initialization.
      mean (np.ndarray): Mean of the dataset.
      components (np.ndarray): Principal components of the dataset.
      explained_variance (np.ndarray): Explained variance of the principal components.
              """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None


    def _get_centered_data(self, dataset: Dataset) -> np.ndarray:
        """
        Center the dataset by subtracting the mean.

        Parameters:
        dataset (Dataset): The dataset to center.

        Returns:
        np.ndarray: A matrix with the centered data.

        """

        self.mean = np.mean(dataset.X, axis=0)  #axis= 0 column, mean of each column
        self.centered_data = dataset.X - self.mean
        return self.centered_data


    def _get_components(self, dataset: Dataset) -> np.ndarray:
        """
        Calculate the principal components of the dataset.

        Parameters:
        dataset (Dataset): The dataset for which to compute principal components.

        Returns:
        np.ndarray: A matrix with the principal components.
        """
        centered_data = self._get_centered_data(dataset)
        self.u_matrix, self.s_matrix, self.v_matrix_t = np.linalg.svd(centered_data, full_matrices=False)
        self.components = self.v_matrix_t[:, :self.n_components]
        return self.components

    def _get_explained_variance(self, dataset: Dataset) -> np.ndarray:
        """
         Calculate the explained variance of the principal components.

        Parameters:
        dataset (Dataset): The dataset to compute explained variance for.

        Returns:
        np.ndarray: A vector with the explained variance.
        """

        ev = self.s_matrix ** 2 / (len(dataset.X) - 1)
        explained_variance = ev[:self.n_components]
        return explained_variance

    def fit(self, dataset: Dataset):
        """
         Fit the PCA model to the provided dataset.

        Parameters:
        dataset (Dataset): The dataset to fit the PCA model to.

        Returns:
        PCA: The PCA instance with calculated components and explained variance.

        """
        if dataset.X is None or dataset.X.shape[0] == 0:
            raise ValueError("Input dataset is empty or has no valid shape.")
        self.components = self._get_components(dataset)
        self.explained_variance = self._get_explained_variance(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the dataset using the fitted PCA model.

        Parameters:
        dataset (Dataset): The dataset to be transformed.

        Returns:
        Dataset: Transformed dataset based on the PCA model.

        """
        if self.components is None:
            raise Exception("You must fit the PCA before transform the dataset.")
        v_matrix = self.v_matrix_t.T
        transformed_data = np.dot(self.centered_data, v_matrix)
        return Dataset(transformed_data, dataset.y, dataset.features, dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit the PCA model to the provided dataset and then transform it.

        Parameters:
        dataset (Dataset): The dataset to be processed.

        Returns:
        Dataset: Transformed dataset based on the fitted PCA model.

        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == '__main__':
    # Test your PCA class
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    features = ['feature1', 'feature2', 'feature3']
    label = 'target'
    dataset = Dataset(X, y, features, label)

    n_components = 2
    pca = PCA(n_components)

    # Fit and transform the dataset
    pca.fit_transform(dataset)

    # Print the components and explained variance
    print("Principal Components:")
    print(pca.components)
    print("Explained Variance:")
    print(pca.explained_variance)




