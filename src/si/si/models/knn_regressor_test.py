import pandas as pd
from sklearn.model_selection import train_test_split
from metrics.rmse import rmse
from models.knn_regressor import KNNRegressor


data = pd.read_csv('cpu.csv')


X = data.drop('target_column_name', axis=1) 
y = data['target_column_name']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_regressor = KNNRegressor(n_neighbors=5)
knn_regressor.fit(X_train, y_train)
y_pred = knn_regressor.predict(X_test)
rmse_value = rmse(y_pred, y_test)
print("RMSE:", rmse_value)
