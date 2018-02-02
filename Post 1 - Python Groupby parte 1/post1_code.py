#!/usr/bin/env python.
"""
File name: post_defaultDict.py
Author: Koldo Pina
Date created: 31/01/2018
Date last modified: 31/01/2018
Python Version: 3.5
"""

#Agrupando con Pandas
import pandas as pd

df = pd.read_csv('trabajos_01.csv', sep=";", decimal=",")
grouped = df.groupby(['DIA_MES'])
res = grouped['HORAS_TRABAJO'].sum()
res_df = pd.DataFrame({'DI_MES': res.index, 'HORAS_TRABAJO': res.values})

print (df.head())
print (df.tail())