#!/usr/bin/env python.
"""
File name: post1_code.py
Author: Koldo Pina
Date created: 04/02/2018
Date last modified: 04/02/2018
Python Version: 3.5
"""

#Agrupando con Pandas
import pandas as pd

df = pd.read_csv('trabajos_02.csv', sep=";", decimal=",")
grouped_df = df.groupby(['ZONA', 'PLANTA', 'DIA_MES']).sum().reset_index()
print(grouped_df)

#Agrupando sin Pandas
#Leemos el csv y creamos una lista con las filas
import csv
from itertools import groupby
from operator import itemgetter

csv_rdr = csv.reader(open('trabajos_02.csv'), delimiter=';')

trabajos_lista_dict = []
for index, row in enumerate(csv_rdr):
    if index == 0:
        encabezado = row
    else:
        row[3] = float(row[3].replace(",", "."))
        d = {}
        for index, value in enumerate(row):
            key = encabezado[index]
            d[key] = value
        trabajos_lista_dict.append(d)
print (trabajos_lista_dict[0])

#Utilizando groupby
result = []
grouper = itemgetter("ZONA", "PLANTA", "DIA_MES")
for key, grp in groupby(sorted(trabajos_lista_dict, key = grouper), grouper):
    temp_dict = dict(zip(["ZONA", "PLANTA", "DIA_MES"], key))
    temp_dict["HORAS_TRABAJO"] = sum(item["HORAS_TRABAJO"] for item in grp)
    result.append(temp_dict)

print ("{:<7}{:<9}{:<11}{:<13}".format("ZONA","PLANTA","DIA_MES","HORAS_TRABAJO"))
for element in result:
    zona, planta, dia_mes, horas_trabajo = element['ZONA'],element['PLANTA'],element['DIA_MES'],element['HORAS_TRABAJO']
    print ("{:<7}{:<9}{:<11}{:<13}".format(zona, planta, dia_mes, str(horas_trabajo)))

#Utilizando defaultdict
from collections import defaultdict

res_dd = []
d = defaultdict(list)
for row in trabajos_lista_dict:
    d[row['ZONA'] + '|' + row['PLANTA'] + '|' + row['DIA_MES']].append(float(row['HORAS_TRABAJO']))
for k, v in sorted(d.items()):
    res_dd.append([k, sum(v)])

print("{:<7}{:<9}{:<11}{:<13}".format("ZONA", "PLANTA", "DIA_MES", "HORAS_TRABAJO"))
for row in res_dd:
    zona, planta, dia_mes = row[0].split('|')
print("{:<7}{:<9}{:<11}{:<13}".format(zona, planta, dia_mes, str(row[1])))
