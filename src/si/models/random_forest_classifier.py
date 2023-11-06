import numpy as np
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from typing import Union


class RandomForestClassifier:
    def __init__(self, n_estimators: int, max_features: Union[int, None] = None,
                 min_samples_split: int = 2, max_depth: Union[int, None] = None,
                 mode: str = "gini", seed: Union[int, None] = None):
        """
        Random Forest Classifier.

        Args:
            n_estimators (int): The number of decision trees in the forest.
            max_features (int or None, optional): The maximum number of features to consider for splitting.
                If None, it defaults to sqrt(n_features). Default is None.
            min_samples_split (int, optional): The minimum number of samples required to split a node.
                Default is 2.
            max_depth (int or None, optional): The maximum depth of the decision trees. Default is None.
            mode (str, optional): The criterion used for splitting. "gini" or "entropy". Default is "gini".
            seed (int or None, optional): The random seed for reproducibility. Default is None.
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """
        Fit the RandomForestClassifier to the training data.

        Args:
            dataset (Dataset): The training dataset used to fit the model.

        Returns:
            RandomForestClassifier: The fitted RandomForestClassifier object.
    """
        if self.seed is not None:
            np.random.seed(self.seed)

        n_samples, n_features = dataset.X.shape
        self.max_features = int(np.sqrt(n_features)) if self.max_features is None else self.max_features

        for _ in range(self.n_estimators):
            # Create a bootstrap dataset
            indices = np.random.choice(n_samples, n_samples, replace=True)
            features = np.random.choice(n_features, self.max_features, replace=False)
            bootstrap_dataset = Dataset(X=dataset.X[indices][:, features],
                                        y=dataset.y[indices],
                                        features=dataset.features[features],
                                        label=dataset.label)

            # Create and train a decision tree with the bootstrap dataset
            tree = DecisionTreeClassifier(min_sample_split=self.min_samples_split, max_depth=self.max_depth,
                                          mode = self.mode)
            tree.fit(bootstrap_dataset)

            # Append a tuple containing the features used and the trained tree
            self.trees.append((features, tree))

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
         Predict the class labels for input data.

        Args:
            dataset (Dataset): The dataset containing input data for prediction.

        Returns:
            np.ndarray: Predicted class labels.
        """
        predictions = []
        for x in dataset.X:
            predictions_tree = []
            for features, tree in self.trees:
                x_subset = x[features]
                # Create a new dataset with the same structure as the original
                subset_dataset = Dataset(X=[x_subset], y=None, features=dataset.features[features], label=dataset.label)
                predictions_tree.append(tree.predict(subset_dataset))
            most_common_prediction = max(set(predictions_tree), key=predictions_tree.count)
            predictions.append(most_common_prediction)

        return np.array(predictions)

    def score(self, dataset: Dataset) -> float:
        """
        Calculate the accuracy of the model on the test data.

        Args:
            dataset (Dataset): The dataset containing test data and true labels.

        Returns:
            float: The accuracy score of the model on the test data.
    """
        y_pred = self.predict(dataset)
        return accuracy(dataset.y, y_pred)
