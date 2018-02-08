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


def groupby_agg(data, groupby_fields, agg_fields, aggr_list, how_list):
    """
    :param data: dataset de datos formato lista de diccionarios
    :param groupby_fields: campo sobre el que agregar. Ejemplo: ['ZONA,'PLANTA',TIPO']
    :param agg_fields: lista con los campos a agregar. Ejemplo: ['EQUIPO', 'KW_FRIO']
    :param aggr_list: lista con las funciones de agregacion. Correspondientes con agg_fields. Ejemplo: ['count', 'sum']
    :param how_list: lista tipo ['distinct', '']
    :return: diccionario -->d_res
    """
    from collections import defaultdict
    d_res = defaultdict(list)
    for index, agg_field in enumerate(agg_fields):
        if aggr_list[index]=='sum':
            d = defaultdict(list)
            for row in data:
                groupby_field_string =''
                for field in groupby_fields:
                    if groupby_field_string!='':
                        groupby_field_string += '|' + row[field]
                    else:
                        groupby_field_string+=row[field]
                d[groupby_field_string].append(row[agg_field])