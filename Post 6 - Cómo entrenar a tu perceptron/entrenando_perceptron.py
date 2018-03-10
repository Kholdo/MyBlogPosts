#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 10/03/2018
Python Version: 3.5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#Cargamos los datos y los echamos un vistazo
import pandas as pd
df = pd.read_csv('perceptron_data_2018310.csv', sep=';')
print('_'*60 + 'COLUMNS')
print(df.columns.values)
print('_'*60 + 'INFO')
print (df.info())
print('_'*60 + 'DESCRIBE')
print (df.describe().transpose())
print('_'*60 + 'SHAPE')
print (df.shape)
print('_'*60 + 'COUNT VALUE CLASSES')
print (df.loc[:,'type'].value_counts())
print('_'*60 + 'NULL VALUES')
print (df.isnull().sum())
print('_'*60 + 'NULL VALUES BIS')
print(df.isnull().values.any())

X = df.iloc[:, 0:2].values
y = df.iloc[:, 2].values
plt.title("Perceptron data'", fontsize='small')
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, alpha=0.5, edgecolor='c')
plt.show()

#Dividimos el dataframe en train y test
X, y = df.loc[:, ['feature1', 'feature2']].values, df.loc[:,['type']].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

#Creamos la clase para el perceptron
class SimplePerceptron():

    def __init__(self, eta):
        """
        :param eta: tasa de aprendizaje
        """
        self.eta = eta

    def zeta(self, X):
        """
        Calcula el producto de la entrada por su peso
        :param X: datos de entrenamiento con las caracteristicas. Array
        """
        zeta = np.dot(1, self.weights[0]) + np.dot(X, self.weights[1:])
        return zeta

    def predict(self, X):
        """
        Calcula la salida de la neurona teniendo en cuenta la función de activación
        :param X: datos con los que predecir la salida de la neurona. Array
        :return: salida de la neurona
        """
        output = np.where(self.zeta(X) >= 0.0, 1, 0)
        return output

    def fit(self, X, y):
        #Ponemos a cero los pesos
        self.weights = [0] * (X.shape[1] + 1)
        self.errors = []
        self.iteraciones = 0
        while True:
            errors = 0
            for features, expected in zip(X,y):
                delta_weight = self.eta * (expected - self.predict(features))
                self.weights[1:] += delta_weight * features
                self.weights[0] += delta_weight * 1
                errors += int(delta_weight != 0.0)
            self.errors.append(errors)
            self.iteraciones += 1
            if errors == 0:
                break

#Creamos una instancia de la clase
sp = SimplePerceptron(eta=0.1)
#Entrenamos
sp.fit(X_train, y_train)
