import numpy as np
from matplotlib import pyplot as plt

from si.data.dataset import Dataset
#from si.metrics.mse import mse


class RidgeRegressionLeastSquares:
    def __init__(self, l2_penalty=1.0, scale=True):
        """
        Initialize the Ridge Regression model.

        Parameters:
            l2_penalty (float,optional)
            scale (bool,optional)

         Attributes:
            l2_penalty (float): The specified L2 regularization parameter.
            scale (bool): Indicates whether the data is scaled.
            theta (numpy.ndarray): Coefficients of the model for every feature.
            theta_zero (float): The zero coefficient (y-intercept) of the model.
            mean (numpy.ndarray, optional): Mean of the dataset for every feature. Used when scale is True.
            std (numpy.ndarray, optional): Standard deviation of the dataset for every feature. Used when scale is True.
        """
        self.l2_penalty = l2_penalty
        self.scale = scale
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, X, y):
        """Fit the Ridge Regression model to the training data.

        Parameters:
         X (array-like): Training data features.
         y (array-like): Target values.
          Returns:
              None
    """
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = (X - self.mean) / self.std

        m, n = X.shape
        I = np.identity(n)
        X_transpose = X.T
        self.theta = np.linalg.inv(X_transpose.dot(X) + self.l2_penalty * I).dot(X_transpose).dot(y)
        self.theta_zero = np.mean(y) - np.dot(self.theta, np.mean(X, axis=0))

    def predict(self, X):
        """
        Predict the dependent variable (y) using the estimated coefficients.
        Parameters:
             X (array-like): Input data features for prediction.
        Returns:
            y_pred (numpy.ndarray): Predicted values of the dependent variable (y).

        """
        if self.scale:
            X = (X - self.mean) / self.std
        return np.dot(X, self.theta) + self.theta_zero

    def score(self, X, y):
        """
        Calculate the error between the actual and predicted values.
        Parameters:
             X (array-like): Input data features for prediction.
             y (array-like): Actual target values.
        Returns:
            mse (float): Mean Squared Error (MSE) between actual and predicted values.

        """
        y_pred = self.predict(X)
        error = y - y_pred
        return np.mean(error**2)

# Example usage:
if __name__ == "__main__":
    # Sample data
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([2, 3, 4, 5])

    # Initialize and fit the RidgeRegressionLeastSquares model
    model = RidgeRegressionLeastSquares(l2_penalty=0.1, scale=True)
    model.fit(X_train, y_train)

    # Predict and score
    X_test = np.array([[5, 6]])
    prediction = model.predict(X_test)
    mse = model.score(X_train, y_train)

    print("Predicted value:", prediction)
    print("Mean Squared Error:", mse)


