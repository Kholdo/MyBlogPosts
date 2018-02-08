#!/usr/bin/env python.
"""
File name: post3.py
Author: Koldo Pina
Date created: 08/02/2018
Date last modified: 08/02/2018
Python Version: 3.5
"""

# Cargamos los datos en una lista de diccionarios
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
        if aggr_list[index] == 'sum':
            d = defaultdict(list)
            for s_row in data:
                groupby_field_string = ''
                for field in groupby_fields:
                    if groupby_field_string != '':
                        groupby_field_string += '|' + s_row[field]
                    else:
                        groupby_field_string += s_row[field]
                d[groupby_field_string].append(s_row[agg_field])
            for k, v in sorted(d.items()):
                if v is None:
                    v = 0.0
                d_res[k].append(sum(v))
        if aggr_list[index] == 'count':
            d = defaultdict(list)
            for c_row in data:
                groupby_field_string = ''
                for field in groupby_fields:
                    if groupby_field_string != '':
                        groupby_field_string += '|' + c_row[field]
                    else:
                        groupby_field_string += c_row[field]
                d[groupby_field_string].append(c_row[agg_field])
            for k, v in sorted(d.items()):
                if how_list[index] == 'distinct':
                    d_res[k].append(len(set([item for item in v if item])))
                else:
                    d_res[k].append(len([item for item in v if item]))
    return d_res


data = kw_cl_list
groupby_fields = ['ZONA', 'PLANTA', 'TIPO']
agg_fields = ['EQUIPO', 'KW_FRIO']
aggr_list = ['count', 'sum']
how_list = ['distinct', '']

groupedData = groupby_agg(data, groupby_fields,
                          agg_fields, aggr_list, how_list)


print("{:<7}{:<9}{:<11}{:<15}{:<20}".format(
    "ZONA", "PLANTA", "TIPO", "NUM_EQUIPOS", "SUMA KW FRIO"))
for key in sorted(groupedData.keys(), key=lambda key: key.split('|')[0]):
    zona, planta, tipo = key.split('|')
    print("{:<7}{:<9}{:<11}{:<15}{:<20}".format(
        zona, planta, tipo, groupedData[key][0], groupedData[key][1]))
