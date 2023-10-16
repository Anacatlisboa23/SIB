import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Retorna o Mean Squared Error do modelo num dado dataset.

    :param y_true: np.ndarray
        Labels reais do dataset
    :param y_pred: np.ndarray
        Labels estimadas do dataset

    :return: mse: float
        O Mean Squared Error do modelo entre y_true e y_pred
    """
    return np.sum((y_true - y_pred) ** 2) / (len(y_true) * 2)


def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Retorna o Meaan Squared Error derivativo do modelo para o y_pred.

    :param y_true: np.ndarray
        Labels reais do dataset
    :param y_pred: np.ndarray
        Labels estimadas do dataset

    :return: mse_derivative: np.ndarray
        O derivative Mean Squared Error
    """
    return -2 * (y_true - y_pred) / (len(y_true) * 2)