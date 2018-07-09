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


## We normalize and divide the data into training data and test data.
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

## We created the Adaline class
