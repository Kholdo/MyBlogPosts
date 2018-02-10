#!/usr/bin/env python.
"""
Author: Koldo Pina
Date created: 08/02/2018
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


class myGroupBy():

    def __init__(self, data, groupby_fields, agg_fields, aggr_list):
        self.data = data
        self.groupby_fields = groupby_fields
        self.agg_fields = agg_fields
        self.aggr_list = aggr_list

        def groupby_agg(self):
            """
            :param data: dataset de datos formato lista de diccionarios
            :param groupby_fields: campo sobre el que agregar. Ejemplo: ['ZONA,'PLANTA',TIPO']
            :param agg_fields: lista con los campos a agregar. Ejemplo: ['EQUIPO', 'KW_FRIO']
            :param aggr_list: lista con las funciones de agregacion. Correspondientes con agg_fields. Ejemplo: ['count', 'sum']
            :return: diccionario
            """

            def aggr_sum(agg_field):
                """
                :param  agg_field: campo que se va a agregar con la funcion sum
                :return diccionario con las claves y los valores agrupados
                """

