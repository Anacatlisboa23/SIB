import numpy as np
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from typing import Literal


class RandomForestClassifier:
    """
    Ensemble machine learning technique that combines multiple decision trees to improve prediction accuracy
    and reduce overfitting
    """

    def __init__(self, n_estimators=100, max_features=None, min_sample_split=2, max_depth=10, mode='gini', seed=None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Fits the random forest classifier to a dataset.
        Train the decision trees of the random forest.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        n_samples, n_features = dataset.shape()
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        for _ in range(self.n_estimators):
            bootstrap_samples = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_features = np.random.choice(n_features, self.max_features, replace=False)
            random_dataset = Dataset(dataset.X[bootstrap_samples][:, bootstrap_features], dataset.y[bootstrap_samples])

            tree = DecisionTreeClassifier(
                min_sample_split=self.min_sample_split,
                max_depth=self.max_depth,
                mode=self.mode
            )
            tree.fit(random_dataset)
            self.trees.append((bootstrap_features, tree))

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predicts the class labels for a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset for which to make predictions.

        Returns
        -------
        np.ndarray
            An array of predicted class labels.
        """
        n_samples = dataset.shape()[0]
        predictions = np.zeros((self.n_estimators, n_samples), dtype=object)

        # for each tree
        row = 0
        for features, tree in self.trees:
            tree_preds = tree.predict(dataset)
            predictions[row, :] = tree_preds
            row += 1

        def majority_vote(sample_predictions):
            unique_classes, counts = np.unique(sample_predictions, return_counts=True)
            most_common = unique_classes[np.argmax(counts)]
            return most_common

        majority_prediction = np.apply_along_axis(majority_vote, axis=0, arr=predictions)

        return majority_prediction

    def score(self, dataset: Dataset) -> float:
        """
        Calculates the accuracy of the model on a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to calculate the accuracy on.

        Returns
        -------
        float
            The accuracy of the model on the dataset.
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)


if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    filename = r"C:\Users\pc\PycharmProjects\si\datasets\iris\iris.csv"
    data = read_csv(filename, sep=",", features=True, label=True)
    train, test = train_test_split(data, test_size=0.33, random_state=42)
    model = RandomForestClassifier(n_estimators=10000, max_features=4, min_sample_split=2, max_depth=5, mode='gini',
                                   seed=42)
    model.fit(train)
    print(model.score(test))


