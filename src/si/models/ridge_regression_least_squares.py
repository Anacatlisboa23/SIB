import numpy as np
from matplotlib import pyplot as plt

from si.data.dataset import Dataset
from si.metrics.mse import mse
class RidgeRegressionLeastSquares:

    def __init__(self, l2_penalty: float = 1, alpha: float = 0.001, max_iter: int = 1000):
        """

        """
        #  parameters:
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter

        # attributes
        self.theta = None
        self.theta_zero = None
        self.cost_history = {}

    def fit(self, dataset: Dataset) -> 'RidgeRegression':
        """
        Faz o fit do modelo

        :param dataset: Dataset
            The dataset to fit the model to

        :return: self: RidgeRegression
            The fitted model
        """
        m, n = dataset.shape()

        # initialize the model parameters
        self.theta = np.zeros(n)  # o tamanho é o número de features (temos um theta para cada uma das features)
        self.theta_zero = 0

        # threshold to stop gradient descent when cost function value stabilizes
        threshold = 1

        # gradient descent
        for i in range(self.max_iter):
            # computing cost and updating cost_history
            self.cost_history[i] = self.cost(dataset=dataset)

            if i > 1 and (self.cost_history[i - 1] - self.cost_history[i] < threshold):
                break
            else:
                # predicted y
                y_pred = np.dot(dataset.X, self.theta) + self.theta_zero  # y=mx+b

                # computing and updating the gradient of the cost function with the learning rate
                gradient = (self.alpha * (1 / m)) * np.dot(y_pred - dataset.y,
                                                           dataset.X)  # np.dot soma os valores das
                # colunas dos arrays de multiplicação
                # o learning rate é multiplicado por 1/m para normalizar o learning rate ao tamanho do dataset

                # computing the l2 penalty
                penalization_term = self.alpha * (self.l2_penalty / m) * self.theta

                # updating the model parameters
                self.theta = self.theta - gradient - penalization_term
                self.theta_zero = self.theta_zero - (self.alpha * (1 / m)) * np.sum(y_pred - dataset.y)

        return self

    def predict(self, dataset: Dataset) -> np.array:
        """
        Faz a previsão do output do dataset

        :param dataset: Dataset
            The dataset to predict the output of

        :return: predictions: np.array
            The predictions of the dataset
        """
        return np.dot(dataset.X, self.theta) + self.theta_zero

    def score(self, dataset: Dataset) -> float:
        """
        Calcula o Mean Square Error do modelo no dataset

        :param dataset: Dataset
            The dataset to compute the MSE on

        :return: mse: float
            The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)
        return mse(dataset.y, y_pred)

    def cost(self, dataset: Dataset) -> float:
        """
        Calcula a cost function (J function) do modelo no dataset, usando a L2 regularization

        :param dataset: Dataset
            The dataset to compute the cost function on

        :return: cost: float
            The cost function of the model
        """
        y_pred = self.predict(dataset)
        return (np.sum((y_pred - dataset.y) ** 2) + (self.l2_penalty * np.sum(self.theta ** 2))) / (2 * len(dataset.y))

    def plot_cost_history(self):
        """
        Faz o plot do cost history em função do nº de iterações.
        """
        plt.plot(self.cost_history.keys(), self.cost_history.values(), "-k")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()

if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset

    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegressionLeastSquares(l2_penalty=1, alpha=0.001, max_iter=1000)
    model.fit(dataset_)

    # get coefs
    print(f"Parameters: {model.theta}")

    # compute the score
    score = model.score(dataset_)
    print(f"Score: {score}")

    # compute the cost
    cost = model.cost(dataset_)
    print(f"Cost: {cost}")

    # predict
    y_pred_ = model.predict(Dataset(X=np.array([[3, 5]])))
    print(f"Predictions: {y_pred_}")

