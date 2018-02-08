#!/usr/bin/env python.
"""
File name: post3.py
Author: Koldo Pina
Date created: 08/02/2018
Date last modified: 08/02/2018
Python Version: 3.5
"""

#Cargamos los datos en una lista de diccionarios
import csv

csv_rdr = csv.reader(open('potencias_frio.csv'), delimiter=';')
kw_cl_list = []
for index, row in enumerate(csv_rdr):
    if index == 0:
        encabezado = row
    else:
        row[4] = int(row[4])
        d = {}
        for index, value in enumerate(row):
            key = encabezado[index]
            d[key] = value
        kw_cl_list.append(d)
