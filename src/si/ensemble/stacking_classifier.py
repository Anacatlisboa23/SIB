import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from typing import Callable

class StackingClassifier:
    """
    Initialize an ensemble classifier that combines the predictions of multiple base classifiers to make a final prediction.
    This is achieved by training a second-level "meta-classifier" to make the final prediction using the outputs of the base classifiers as input.

    Parameters:
    - models (list): A collection of different models for the ensemble.
    - final_model: The model responsible for generating the final prediction.

    Attributes:
    - models (list): A collection of different models used within the ensemble.
    - final_model: The model responsible for delivering the ultimate prediction.

    """

    def __init__(self, models: list, final_model):
        """
       Initialize an ensemble stacking classifier.

        Parameters:
        - models (list): Different models for the ensemble.
        - final_model: The final model used to produce the ultimate prediction.
        """
        # parameters
        self.models = models
        self.final_model = final_model

    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the models to the dataset.

        Parameters:
        - dataset (Dataset): The dataset object to which the models are fitted.

        Returns:
        - self (StackingClassifier): Returns the StackingClassifier instance after fitting the models.
        """
        # training the models
        for model in self.models:
            model.fit(dataset)

        # getting the models' predictions
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))

        # training the final model
        self.final_model.fit(Dataset(dataset.X, np.array(predictions).T))

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Compute the predictions of all the models and return the final model prediction.

        Parameters:
        - dataset (Dataset): The dataset object for which to predict the labels.

        Returns:
        - y_pred (np.array): The final model prediction.
        """
        # gets the model predictions
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))

        # gets the final model predictions
        y_pred = self.final_model.predict(Dataset(dataset.X, np.array(predictions).T))

        return y_pred

    def score(self, dataset: Dataset) -> float:
        """
        Calculate the accuracy of the model.

        Returns:
        - score (float): The accuracy of the model.
        """
        y_pred = self.predict(dataset)
        score = accuracy(dataset.y, y_pred)

        return score



