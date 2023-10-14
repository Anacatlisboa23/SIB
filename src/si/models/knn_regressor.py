from typing import Callable, Union
import numpy as np
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor:

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initializes the KNN classifier
        Parameters
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        # parameters
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        Fits the model to the given dataset
        Parameters:

    dataset: Dataset

        Returns:

        self: KNNClassifier
            The fitted model
        """
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the closest label of the given sample
        Parameters:
        sample: np.ndarray
            The sample to get the closest label of
        Returns:
        label: str or int
            The closest label
        """
        # compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)
        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]
        # get the labels of the k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]
        tmp = np.average(k_nearest_neighbors_labels)

        return tmp

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the classes of the given dataset
        Parameters:
        dataset: Dataset
        Returns:
        predictions: np.ndarray

        """
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """

        Parameters:
        dataset: Dataset

        Returns:
        accuracy: float
        """
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    import pandas as pd
    from sklearn import preprocessing

    # dataset_ = Dataset.from_random(100, 5)
    df = pd.read_csv(r"C:\Users\anali\PycharmProjects\si\datasets\cpu\cpu.csv")
    print(df.head())
    dataset_ = Dataset.from_dataframe(df, label='perf')
    # dataset_.X = preprocessing.scale(dataset_.X)

    # load and split the dataset
    # dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.25)

    # initialize the KNN classifier
    knn = KNNRegressor(k=5)

    # fit the model to the train dataset
    knn.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn.score(dataset_test)
    print(f'The rmse of the model is: {score}')