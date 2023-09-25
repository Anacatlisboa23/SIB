# exercicios 1:

import numpy as np
import pandas as pd


ficheiro ='C:\\Users\\catarina\\si\\datasets\\iris\\iris.csv'
df = pd.read_csv(ficheiro)
print(df.head())

penultimate_variable = df.iloc[:, -2]
dimension = penultimate_variable.shape
print(dimension)

# dimensão (150,) é uma estrutura unidimensional (um array ou uma série) e possui 150 elementos numa única dimensão. 

last10samples=df.tail(10)
means = last10samples.mean(numeric_only=True)
print (means)

mask = (df.iloc[:, :-1] <= 6).all(axis=1)
selected_samples = df[mask]
num_samples = selected_samples.shape[0]
print(num_samples) #89 amostras

filtered_samples = df[df['class'] != 'Iris-setosa']
num_saples_filtered= filtered_samples.shape[0]
print(num_saples_filtered) #100

