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
