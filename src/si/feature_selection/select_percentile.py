import numpy as np
from typing import Callable
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectPercentile:
    def __init__(self, score_func: Callable = f_classification, percentile: float = 0.5):
        """
        Initializes the SelectPercentile class.

        Args:
            score_func (Callable): The scoring function to be used for feature selection.
            percentile (float): The percentile of features to be retained after selection.
        """
        self.percentile = percentile
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Fits the SelectPercentile instance to the training data.

        Args:
            dataset (Dataset): The training dataset.

        Returns:
            SelectPercentile: The fitted SelectPercentile instance.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset according to the feature selection.

        Args:
            dataset (Dataset): The dataset to be transformed.

        Returns:
            Dataset: A new dataset containing only the selected features.
        """

        # Calculate the total number of features in the dataset
        num_total = len(list(dataset.features))
        # Calculate the number of features to keep based on the specified percentile
        num_a_manter = int(num_total * self.percentile)
        # Get the indices of the features to keep, sorted by score values
        idxs = np.argsort(self.F)[-num_a_manter:]
        # Create a new array of features containing only the selected features
        features = np.array(dataset.features)[idxs]
        # Create a new dataset containing only the selected features
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)
    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fits the SelectPercentile instance to the training data and then transforms the dataset.

        Args:
            dataset (Dataset): The training dataset.

        Returns:
            Dataset: A new dataset containing only the selected features.
        """
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    import pandas as pd
    import numpy as np

    # Load the dataset using pandas
    df = pd.read_csv(r"C:\Users\pc\PycharmProjects\si\datasets\iris\iris.csv")

    # Extract features and labels from the DataFrame
    X = df.drop(columns=["class"]).values
    y = df["class"].values

    # Create a Dataset object
    dataset = Dataset(X=X, y=y, features=list(df.columns[:-1]), label='class')

    # Create a SelectPercentile instance
    selector = SelectPercentile(percentile=0.5)

    # Fit and transform the dataset
    selected_dataset = selector.fit_transform(dataset)

    # Display the selected features
    print("Selected features:")
    print(selected_dataset.features)
