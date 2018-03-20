#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 18/03/2018
Python Version: 3.5
"""

import pandas as pd

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


