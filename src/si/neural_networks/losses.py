from abc import abstractmethod

import numpy as np


class LossFunction:

    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Compute the derivative of the loss function for a given prediction.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """
    Mean squared error loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        return np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the mean squared error loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # To avoid the additional multiplication by -1 just swap the y_pred and y_true.
        return 2 * (y_pred - y_true) / y_true.size


class BinaryCrossEntropy(LossFunction):
    """
    Cross entropy loss function.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        float
            The loss value.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the cross entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true labels.
        y_pred: numpy.ndarray
            The predicted labels.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        # Avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - (y_true / p) + (1 - y_true) / (1 - p)

#Exercise 14
class CategoricalCrossEntropy(LossFunction):
    """
    Categorical cross-entropy loss function.

    Measures the dissimilarity between predicted class probabilities
    and true one-hot encoded class labels.
    """

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the categorical cross-entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true one-hot encoded class labels.
        y_pred: numpy.ndarray
            The predicted class probabilities.

        Returns
        -------
        float
            The loss value.
        """
        epsilon = 1e-15  # Small constant to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
        return loss

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the derivative of the categorical cross-entropy loss function.

        Parameters
        ----------
        y_true: numpy.ndarray
            The true one-hot encoded class labels.
        y_pred: numpy.ndarray
            The predicted class probabilities.

        Returns
        -------
        numpy.ndarray
            The derivative of the loss function.
        """
        epsilon = 1e-15  # Small constant to avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / y_pred / len(y_true)

#test
from si.neural_networks.losses import CategoricalCrossEntropy

if __name__ == "__main__":
    # Create an instance of CategoricalCrossEntropy
    categorical_cross_entropy = CategoricalCrossEntropy()

    # Test CategoricalCrossEntropy
    print("CategoricalCrossEntropy:")
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # One-hot encoded labels
    y_pred = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])  # Predicted probabilities

    loss_value = categorical_cross_entropy.loss(y_true, y_pred)
    derivative_value = categorical_cross_entropy.derivative(y_true, y_pred)

    print("True Labels:")
    print(y_true)
    print("Predicted Probabilities:")
    print(y_pred)
    print("Loss Value:")
    print(loss_value)
    print("Derivative Value:")
    print(derivative_value)