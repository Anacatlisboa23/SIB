
import sys
import os
import numpy as np
import pandas as pd
sys.path.append('C:\\Users\\anali\\Documents\\GitHub\\si\\src\\si')

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification
from typing import Callable

class SelectPercentile:
    
    def __init__(self, score_func: Callable = f_classification, percentile: float = 0.5): # 0.5 porque queremos manter metade
        """
        Initializes the SelectPercentile class.

        Args:
            score_func (Callable): The scoring function to be used for feature selection. It should be a function that accepts a Dataset object and returns two lists,
            one with scores for each feature and another with p-values.
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

        num_total = len(list(dataset.features))
        num_a_manter = int(num_total * self.percentile)
        idxs = np.argsort(self.F)[-num_a_manter:]
        features = np.array(dataset.features)[idxs]
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
    from si.data.dataset import Dataset

    #df = pd.read_csv("C:\\Users\\catarina\\si\\datasets\\iris\\iris.csv")
    df= pd.read_csv("C:\\Users\\anali\\Documents\\GitHub\\si\\datasets\\iris")
    dataset = Dataset.from_dataframe(df, label='class')
    # dataset = Dataset(X=np.array([[0, 2, 0, 3],
    #                               [0, 1, 4, 3],
    #                               [0, 1, 1, 3]]),
    #                   y=np.array([0, 1, 0]),
    #                   features=["f1", "f2", "f3", "f4"],
    #                   label="y")

    selector = SelectPercentile(percentile=0.5)
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)