#!/usr/bin/env python.
"""
File name: post1_code.py
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
res_df = pd.DataFrame({'DIA_MES': res.index, 'HORAS_TRABAJO': res.values})

print (res_df.head())
print (res_df.tail())


#Agrupando sin Pandas
#Leemos el csv y creamos una lista con las filas
import csv

csv_rdr = csv.reader(open('trabajos_01.csv'), delimiter=';')
trabajos_list = []
for index, row in enumerate(csv_rdr):
    if index == 0:
        encabezado = row
    else:
        row[3] = float(row[3].replace(",", "."))
        trabajos_list.append(row)

#Utilizando groupby
from operator import itemgetter
from itertools import groupby

res = []
trabajos_group = groupby(sorted(trabajos_list, key=itemgetter(2)), itemgetter(2))
for key, grp in trabajos_group:
    res.append([key, sum(ele[3] for ele in grp)])
print (res)

#Utilizando defaultdict
from collections import defaultdict

res_2 = []
d = defaultdict(list)
for row in trabajos_list:
    d[row[2]].append(float(row[3]))
for k, v in sorted(d.items()):
    res_2.append([k, sum(v)])
print (res_2)