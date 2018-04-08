#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 18/03/2018
Python Version: 3.5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import csv

## Modelo "cincel y martillo"

# 1 - Obtenemos los datos
with open('data_LR.csv', 'r') as f:
    reader = csv.reader(f, delimiter='|')
    OUTDOOR_TEMP = []
    ELECTRIC_POWER = []
    for index, row in enumerate(reader):
        if index == 0:
            header = row
        else:
            try:
                outNum = float(row[0].replace(",", "."))
            except:
                outNum = None
            OUTDOOR_TEMP.append(outNum)
            try:
                elecNum = float(row[1].replace(",", "."))
            except:
                elecNum = None
            ELECTRIC_POWER.append(elecNum)

# 2 - Un vistazo a los datos
print('_'*60 + 'COLUMNS')
print (header)

def info(header, data_list):
    """
    :param header: lista con los encabezados de las columnas
    :param data_list: lista con las listas de datos de las columnas: [lista1, lista2, etc...]
    :return: diccionario de diccionarios de valores con número de registros, tipo de dato, número de na, media, std, min, max de cada columna
    """

    from collections import defaultdict

    header = header
    columns = data_list
    values = defaultdict()
    for index, head in enumerate(header):
        aux = defaultdict()
        aux['len'] = len(columns[index])
        aux['clases'] = set([type(ele) for ele in columns[index]])
        aux['na'] = sum(1 for ele in columns[index] if ele == None)
        # media
        media = sum(ele for ele in columns[index] if ele != None) / len(columns[index])
        aux['media'] = media
        # std
        n = sum(1 for ele in columns[index] if ele != None)
        std = ((1 / (n - 1)) * sum((ele - media) ** 2 for ele in columns[index] if ele != None)) ** 0.5
        aux['std'] = std
        # minimo
        aux['min'] = min(ele for ele in columns[index] if ele != None)
        # maximo
        aux['max'] = max(ele for ele in columns[index] if ele != None)
        values[head] = aux
    return values

print('_'*60 + 'INFO')
print (info(header, [OUTDOOR_TEMP, ELECTRIC_POWER]))
#Quitamos los nas
index_to_drop = [index for index, val in enumerate(ELECTRIC_POWER) if val is None]
ELECTRIC_POWER = [val for index, val in enumerate(ELECTRIC_POWER) if index not in index_to_drop]
OUTDOOR_TEMP = [val for index, val in enumerate(OUTDOOR_TEMP) if index not in index_to_drop]
print (info(header, [OUTDOOR_TEMP, ELECTRIC_POWER]))

# 3 - Exploramos y modificamos los datos

def visual(header, X, y):
    """
    :param header: Lista con los nombres de los encabezados
    :param X: Lista con los valores de la columna a colocar en el eje X
    :param y: Lista con los valores de la columna a colocar en el eje y
    :return: matplotlib figure plot
    """

    fs = 10  # fontsize
    fig, axs = plt.subplots(3, 2, figsize=(6, 6))
    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.5, wspace=0.2, left=0.125, right=0.9)
    axs[0, 0].scatter(X, y, c='r', edgecolors=(0, 0, 0), alpha=0.2)
    axs[0, 0].set_title('Scatter %s vs %s' %(header[1], header[0]), fontsize=fs)
    axs[1, 0].hist(X, color='red')
    axs[1, 0].set_title('Hist %s' %header[0], fontsize=fs)
    axs[0, 1].hist2d(X, y)
    axs[0, 1].set_title('Hist 2D', fontsize=fs)
    axs[1, 1].hist(y, color='blue')
    axs[1, 1].set_title('Hist %s' %header[1], fontsize=fs)
    axs[2, 0].boxplot(X)
    axs[2, 0].set_title('Box %s' %header[0], fontsize=fs)
    axs[2, 1].boxplot(y)
    axs[2, 1].set_title('Box %s' %header[1], fontsize=fs)
    plt.show()

visual(header, OUTDOOR_TEMP, ELECTRIC_POWER)

#Quitamos los outlier o valores atípicos
index_to_drop = [index for index, val in enumerate(OUTDOOR_TEMP) if val == 0]
ELECTRIC_POWER = [val for index, val in enumerate(ELECTRIC_POWER) if index not in index_to_drop]
OUTDOOR_TEMP = [val for index, val in enumerate(OUTDOOR_TEMP) if index not in index_to_drop]

print (info(header, [OUTDOOR_TEMP, ELECTRIC_POWER]))

visual(header, OUTDOOR_TEMP, ELECTRIC_POWER)

#sns.jointplot(np.asarray(OUTDOOR_TEMP) ,np.asarray(ELECTRIC_POWER), dropna = True, kind="hex")

# 4 - Creamos los datos de entrenamiento y testeo

#Creamos las listas para training y test aleatoriamente
all_data = [[x_val, y_val] for x_val, y_val in zip(OUTDOOR_TEMP, ELECTRIC_POWER)]
print (all_data[:5])
random.shuffle(all_data)
div = math.ceil(len(all_data)*0.3)
data_train = all_data[div:]
data_test = all_data[:div]

data_train_X = [ele[0] for ele in data_train]
data_train_y = [ele[1] for ele in data_train]
data_test_X = [ele[0] for ele in data_test]
data_test_y = [ele[1] for ele in data_test]

# 5 - Creamos el modelo y lo entrenamos
# Creamos la clase para el modelo de regresión lineal simple
class Lin_reg():

    def __init__(self, X, Y):
        """
        :param X: lista con los valores de la variable de las abscisas
        :param y: lista con los valores de la variable de las ordenadas
        """
        self.X = X
        self.y = Y
        self.N = len(self.X)
        self.X_mean = sum(self.X) / len(self.X)
        self.y_mean = sum(self.y) / len(self.y)
        self.X_std = (1 / (self.N - 1) * sum((ele - self.X_mean) ** 2
                                             for ele in self.X)) ** 0.5
        self.y_std = (1 / (self.N - 1) * sum((ele - self.y_mean) ** 2
                                             for ele in self.y)) ** 0.5
        self.X_var = self.X_std ** 2
        self.y_var = self.y_std ** 2
        self.cov = sum([i * j for (i, j) in zip([ele - self.X_mean for ele in self.X],
                                                [ele - self.y_mean for ele in self.y])]) / (self.N)

        self.r = self.cov / (self.X_std * self.y_std)

    def Coeficientes(self):
        if len(self.X) != len(self.y):
            raise ValueError('unequal length')
        self.b = self.cov / self.X_var
        self.a = self.y_mean - (self.b * self.X_mean)
        return self.a, self.b

    def predict(self, X):
        yp = []
        for x in X:
            yp.append(self.a + self.b * x)
        return yp

# Creamos una instancia de la clase pasándole los datos de entrenamiento
mylinreg=Lin_reg(data_train_X,data_train_y)
#Le pedimos al modelo los coeficientes:
a, b = mylinreg.Coeficientes()
print ('La recta de regresión es: y = %f + %f * X'%(mylinreg.Coeficientes()))
print('El coeficiente de correlación es: r = %f' %mylinreg.r)

#Dibujamos los datos de entrenamiento y la recta de regresión
plt.title('training values and regression line', fontsize=10)
plt.scatter(data_train_X, data_train_y, c='r', edgecolors=(0, 0, 0), alpha=0.5)
plt.plot(data_train_X, [a + b * x for x in data_train_X])
plt.xlabel('ºC')
plt.ylabel('kW')
plt.show()

# 6 - Hacemos la predicción con los datos de testeo
predictions = mylinreg.predict(data_test_X)

plt.scatter(data_test_y, predictions, c='r', edgecolors=(0, 0, 0), alpha=0.5)
plt.title('predicted values vs real values', fontsize=10)
plt.xlabel('real values')
plt.ylabel('predicted values')
plt.show()

#Ploteo de valores test y predicted
plt.scatter(data_test_X, data_test_y, c='b', edgecolors=(0, 0, 0), alpha=0.5)
plt.scatter(data_test_X, predictions, c='r', edgecolors=(0, 0, 0), alpha=0.5)
plt.title('test_values and regression line', fontsize=10)
plt.xlabel('Outdoor Temp')
plt.ylabel('Electric Power')
plt.show()

# 7 - Evaluación del modelo
#Metricas
#Mean Error - Desviación media
ME = sum(y_pred - y_test for y_pred, y_test in zip(predictions,data_test_y)) / len(predictions)
#Mean Absolute Error (error absoluto medio)
MAE = sum(abs(y_pred - y_test) for y_pred, y_test in zip(predictions,data_test_y)) / len(predictions)
#Mean Square Error (error cuadrático medio)
MSE = sum((y_pred - y_test)**2 for y_pred, y_test in zip(predictions, data_test_y)) / len(predictions)
#Root Mean Square Error - error de la raíz cuadrada de la media RMSE
RMSE = MSE ** 0.5
#Standard Desviation of Residuals . Desviación típica de los residuos
SDR = (1 / (len(data_test_y) - 1) * sum((y_test - y_pred) ** 2
        for y_pred, y_test in zip(predictions, data_test_y))) ** 0.5
print ('Mean Error: %f' %ME)
print ('Mean Absolute Error: %f' %MAE)
print ('Mean Square Error: %f' %MSE)
print ('Root Mean Square Error: %f' %RMSE)
print ('Standard Desviation of Residuals: %f' %SDR)

#Correlation coefficient R2
data_test_mean = sum(ele
    for ele in data_test_y) / len(data_test_y)
predictions_mean = sum(ele
    for ele in predictions) / len(predictions)

data_test_std = (1 / (len(data_test_y) - 1) * sum((ele - data_test_mean) ** 2
    for ele in data_test_y)) ** 0.5
predictions_std = (1 / (len(predictions) - 1) * sum((ele - predictions_mean) ** 2
    for ele in predictions)) ** 0.5
cov = sum([i * j
    for (i, j) in zip([ele - data_test_mean
        for ele in data_test_y],
                      [ele - predictions_mean
        for ele in predictions])
]) / (len(predictions))

print('El coeficiente de correlación es: R2 = %f' % (cov ** 2 / (data_test_std ** 2 * predictions_std ** 2)))

#Residuos
sns.distplot((np.asarray(data_test_y) - np.asarray(predictions)), bins = 50)
plt.show()


## Modelo scikit-learn

# 1- Importamos las librerias necesarias y los datos en un dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


slr_df = pd.read_csv('data_LR.csv', sep="|")
slr_df = slr_df.replace(',','.', regex=True).astype(float)

# 2- Echamos un vistazo a los datos
print (slr_df.head())
print (slr_df.describe())
print (slr_df.info())
print('_'*30 + 'NULL VALUES')
print (slr_df.isnull().sum())
#Eliminamos los NaN
slr_df = slr_df.dropna()

# 3 - Exploramos y modificamos los datos

sns.jointplot(data = slr_df, x = 'OUTDOOR_TEMP', y ='ELECTRIC_POWER')
plt.show()

slr_df = slr_df[slr_df.OUTDOOR_TEMP != 0]

sns.jointplot(data = slr_df, x = 'OUTDOOR_TEMP', y ='ELECTRIC_POWER')
plt.show()

sns.jointplot(data = slr_df, x = 'OUTDOOR_TEMP', y ='ELECTRIC_POWER', kind = 'hex')
plt.show()

sns.pairplot(slr_df)
plt.show()

sns.lmplot(data = slr_df, x = 'OUTDOOR_TEMP', y ='ELECTRIC_POWER')
plt.show()

# 4 - Creamos los datos de entrenamiento y testeo

X = slr_df['OUTDOOR_TEMP'].values.reshape(-1,1)
y = slr_df['ELECTRIC_POWER'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.3, random_state=10)

# 5 - Creamos el modelo y lo entrenamos.

lm = LinearRegression()
lm.fit(X_train, y_train)

slope = lm.coef_
intercept = lm.intercept_

print ('La recta de regresión es: y = %f + %f * X'%(lm.intercept_, slope))

# 6 - Hacemos la predicción con los datos de test.

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions, c='g', edgecolors=(0, 0, 0), alpha=0.5)
plt.title('predicted values vs test values', fontsize=10)
plt.xlabel('test values')
plt.ylabel('predicted values')
plt.show()

# 7 - Evaluación del modelo
print('MAE: ', metrics.mean_absolute_error(y_test, predictions))
print('MAE cincel: ', MAE)
print('MSE: ', metrics.mean_squared_error(y_test, predictions))
print('MSE cincel: ', MSE)
print('RMSE: ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
print('RMSE cincel: ', RMSE)

print ('explained_variance_score', metrics.explained_variance_score(y_test, predictions))
print ('r2_score', metrics.r2_score(y_test, predictions))

sns.distplot((y_test - predictions)), bins = 50)
plt.show()