#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 05/07/2018
Python Version: 3.6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn import metrics

df_adaline = pd.read_csv('adaline_data.csv', sep=';')
df_perceptron = pd.read_csv('perceptron_data.csv', sep=';')
datasets = [df_adaline, df_perceptron]
datanames = ['df_adaline', 'df_perceptron']

X_a = df_adaline.iloc[:, [0, 1]].values
y_a = df_adaline.iloc[:, 2].values
X_p = df_perceptron.iloc[:, [0, 1]].values
y_p = df_perceptron.iloc[:, 2].values

colors = ['#D7D98E', '#91CFEA']
palette = sns.color_palette(colors)
cmap = ListedColormap(colors)

plt.figure(figsize=(15, 15))
plt.subplot(221)
plt.title('ADALINE DATA')
plt.scatter(X_a[:, 0], X_a[:, 1], marker='o', c=y_a,
            s=25, edgecolor='k', cmap=cmap)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.subplot(222)
plt.title('PERCEPTRON DATA')
plt.scatter(X_p[:, 0], X_p[:, 1], marker='o', c=y_p,
            s=25, edgecolor='k', cmap=cmap)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.subplot(223)
sns.countplot(df_adaline['type'], label='count', palette=palette)
plt.subplot(224)
sns.countplot(df_perceptron['type'], label='count', palette=palette)

plt.show()

def describe_plus(df):
    """
    Function that returns describe dataframe plus number of NaNs, number of uniques values, the mode
    # and the std per column
    :param df: dataframe
    :return: dataframe
    """
    describe = df.describe()
    describe.loc['NaNs'] = [df_adaline[feature].isnull().sum() for feature in df.columns]
    describe.loc['uniques'] = [len(df_adaline[feature].unique()) for feature in df.columns]
    describe.loc['mode'] = [df_adaline[feature].mode()[0] for feature in df.columns]
    describe.loc['std'] = [df_adaline[feature].std() for feature in df.columns]
    return describe.transpose()

print (describe_plus(df_adaline))

print (describe_plus(df_perceptron))


## We normalize and divide the data into training data and test data. And train data visualization.
train_test_data = {}

for index, dataset in enumerate(datasets):
    X = dataset.iloc[:, [0, 1]].values
    y = dataset.iloc[:, 2].values
    # normalizamos
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    # Dividimos los dataset en datos de entrenamiento y de prueba
    X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3)
    # Los añadimos al diccionario
    train_test_data[datanames[index]] = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

    plt.title('%s | Normalizado' % datanames[index])
    plt.scatter(X_std[:, 0], X_std[:, 1], marker='o', c=y,
                s=25, edgecolor='k', cmap=cmap)
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')

    plt.show()

## We created the Adaline class

class Adaline_k():

    def __init__(self, eta=0.0001, n_iter=50):
        """
        :param eta: tasa de aprendizaje
        :param n_iter: número de iteraciones(epochs)
        """
        self.eta = eta
        self.n_iter = n_iter

    def zeta(self, X):
        """
        Calcula el producto de las entradas por sus pesos
        :param X: datos de entrenamiento con las caracteristicas. Array
        """
        res = np.dot(1, self.weights[0]) + np.dot(X, self.weights[1:])
        return res

    def activacion(self, X):
        """
        Función lineal de activacion. En este caso sera la misma que zeta
        """
        return self.zeta(X)

    def fit(self, X, y):
        # Generamos pesos iniciales aleatorios
        self.weights = np.random.random_sample((X.shape[1] + 1,))
        # Creamos dos listas para añanir el valor de la funcion de coste
        # y el número de iteración
        self.iters = []
        self.coste = []
        # Comenzamos las iteraciones (epochs)
        for iter in range(self.n_iter):
            # Calculamos el producto de las entradas por sus pesos, esto es,
            # la función zeta
            zeta = self.zeta(X)
            # Calculamos los errores entre las salidas obtenidas y las esperadas
            errors = (y - zeta)
            # Actualizamos los pesos
            self.weights[1:] += self.eta * X.T.dot(errors)
            self.weights[0] += self.eta * errors.sum()
            # Calculamos el valor de la funcion de coste
            coste = 0.5 * np.power(errors, 2).sum()
            # Guardamos el valor del guardiente de la funcion de coste
            # y tambien guardamos el número de iteración (epoch)
            self.coste.append(coste)
            self.iters.append(iter)
        print (f'Modelo Adaline_k entrenado correctamente:')
        print (f'pesos finales: {self.weights}')
        print (f'coste: {self.coste}')

    def predict(self, X):
        """
        Calcula la salida de la neurona teniendo en cuenta la función de activación
        :param X: datos con los que predecir la salida de la neurona. Array
        :return: salida de la neurona
        """
        return np.where(self.activacion(X) >= 0.0, 1, -1)

def plot_decision_regions(X, y, classifier, X_test, y_test, resolution=0.02, test_idx=False):
    markers = ('s', 'x')
    colors = ['#D7D98E', '#91CFEA']
    palette = sns.color_palette(colors)
    cmap = ListedColormap(colors)
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
            alpha=1, c=cmap(idx),
            marker=markers[idx], label='%s | train set'%str(cl), linewidths=0.5,edgecolors= 'grey')
    if test_idx:
        cmap_test = ListedColormap(colors)
        for idx, cl in enumerate(np.unique(y_test)):
            plt.scatter(x=X_test[y_test == cl, 0], y=X_test[y_test == cl, 1],
                alpha=1, c=cmap_test(idx), linewidths=0.8,
                marker='o', label='%s | test set'%str(cl), edgecolors= 'black')

# TRAINING AND METRICS
for index, dataset in enumerate(datasets):
    dataname = datanames[index]
    print(datanames[index] + ' | ' + '#' * 50)
    X = train_test_data[datanames[index]]['X_train']
    y = train_test_data[datanames[index]]['y_train']
    X_test = train_test_data[datanames[index]]['X_test']
    y_test = train_test_data[datanames[index]]['y_test']

    adaGDK = Adaline_k(n_iter=15, eta=0.001)

    adaGDK.fit(X, y)

    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    plot_decision_regions(X, y, classifier=adaGDK, X_test=X_test, y_test=y_test, test_idx=True)
    plt.title(f'Adaline - Gradient Descent K - {dataname}')
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.legend(loc='upper right')

    plt.subplot(122)
    plt.plot(range(1, len(adaGDK.coste) + 1), adaGDK.coste, marker='o')
    plt.title(f'Función de coste vs Epoch - {dataname}')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.tight_layout()
    plt.show()

    print('EVALUANDO MODELO | ' + '#' * 50)
    print('MATRIZ DE CONFUSION')
    predicciones = adaGDK.predict(X_test)
    cm = metrics.confusion_matrix(y_test, predicciones)
    recall_score = metrics.recall_score(y_test, predicciones)
    fig = plt.figure(figsize=(10, 6))
    cm_char = fig.add_subplot(1, 1, 1)
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Pastel2_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.setp(cm_char.get_xticklabels(), visible=False)
    plt.setp(cm_char.get_yticklabels(), visible=False)
    plt.tick_params(axis='both', which='both', length=0)
    title = f'{dataname} | Recall Score: {recall_score}'
    plt.title(title, size=15)
    plt.show()

    print(f'Sensibilidad (Recall): {recall_score}')
    acc_score = metrics.accuracy_score(y_test, predicciones)
    print(f'Acccuracy score: {acc_score}')
    precision = metrics.precision_score(y_test, predicciones)
    print(f'Precision score: {precision}')