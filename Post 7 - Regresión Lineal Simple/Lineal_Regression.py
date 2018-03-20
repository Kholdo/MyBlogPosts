#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 18/03/2018
Python Version: 3.5
"""

import pandas as pd
import matplotlib.pyplot as plt

#Loading data into a dataframe
df = pd.read_csv('data_LR.csv', sep="|")
df = df.replace(',','.', regex=True).astype(float)
#dataframe info section
print(df.head())
print(df.tail())
print('_'*60 + 'COLUMNS')
print(df.columns.values)
print('_'*60 + 'SHAPE')
print(df.shape)
print('_'*60 + 'INFO')
print(df.info())
print('_'*60 + 'DESCRIBE')
print(df.describe())
df = df[df['KWCLT']>0]
df = df[df['KWIT']>0]
print(df.describe())
print('_'*60 + 'NULL VALUES')
print(df.isnull().sum())

#drawing the data
X, y = df['TEMPEXT'].values, df['KWCLT'].values

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axs[0, 0].scatter(X, y, c='r', edgecolors=(0, 0, 0))
axs[0, 0].set_title('Scatter X y')
axs[1, 0].hist(X, color='red')
axs[1, 0].set_title('Hist X')
axs[0, 1].hist(y, color='blue')
axs[0, 1].set_title('Hist y')
axs[1, 1].hist2d(X, y)
axs[1, 1].set_title('Hist 2D')
plt.show()


