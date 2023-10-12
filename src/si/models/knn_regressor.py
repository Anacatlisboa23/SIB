# models/knn_regressor.py

from sklearn.neighbors import KNeighborsRegressor

class KNNRegressor:
    def __init__(self, n_neighbors=5):
        """
        Initialize the KNN Regressor.

        Parameters:
        - n_neighbors: int, number of neighbors to consider (default: 5).
        """
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)

    def fit(self, X, y):
        """
        Fit the KNN Regressor to the training data.

        Parameters:
        - X: array-like, shape (n_samples, n_features), training data.
        - y: array-like, shape (n_samples,), target values.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict target values for a given set of input data.

        Parameters:
        - X: array-like, shape (n_samples, n_features), input data.

        Returns:
        - y_pred: array-like, shape (n_samples,), predicted target values.
        """
        return self.model.predict(X)
