#exercicio 2
import numpy as np

class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def dropna(self):
        nan_mask = np.isnan(self.X).any(axis=1)
        self.X = self.X[~nan_mask]
        self.y = self.y[~nan_mask]

    def fillna(self, value):
        if value == "mean":
            fill_values = np.nanmean(self.X, axis=0)
        elif value == "median":
            fill_values = np.nanmedian(self.X, axis=0)
        else:
            fill_values = value

        for i in range(self.X.shape[1]):
            nan_indices = np.isnan(self.X[:, i])
            self.X[nan_indices, i] = fill_values[i]

    def remove_by_index(self, index):
        if index < 0 or index >= len(self.X):
            raise ValueError("Invalid index")

        self.X = np.delete(self.X, index, axis=0)
        self.y = np.delete(self.y, index)

# Examplo:
# Creação do Dataset 
X = np.array([[1, 2, 3], [4, 5, 6], [7, np.nan, 9]])
y = np.array([0, 1, 2])
dataset = Dataset(X, y)

# Exercicio 2.1: 
dataset.dropna()
print("dropna:")
print(dataset.X)
print(dataset.y)

# Exercicio 2.2: 
dataset = Dataset(X, y)  
dataset.fillna("mean")
print("fillna with mean:")
print(dataset.X)

# Exercicio 2.3: 
dataset = Dataset(X, y)  
index_to_remove = 1  
dataset.remove_by_index(index_to_remove)
print("removing sample by index:")
print(dataset.X)
print(dataset.y)

