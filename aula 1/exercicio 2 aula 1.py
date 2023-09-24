#exercicio 2 aula 1
import numpy as np

class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def dropna(self):
        # Find rows with NaN values in the feature matrix X
        nan_rows = np.isnan(self.X).any(axis=1)

        # Remove rows with NaN values from the feature matrix X and update y accordingly
        self.X = self.X[~nan_rows]
        self.y = self.y[~nan_rows]

        return self

# Example usage:
# Create a Dataset object
X = np.array([[1.0, 2.0], [3.0, np.nan], [4.0, 5.0]])
y = np.array([0, 1, 2])
dataset = Dataset(X, y)

# Use the dropna method to remove rows with NaN values
dataset.dropna()

# Check the modified dataset
print("Modified X:")
print(dataset.X)
print("Modified y:")
print(dataset.y)
