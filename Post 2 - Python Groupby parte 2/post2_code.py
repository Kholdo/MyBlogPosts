#!/usr/bin/env python.
"""
File name: post1_code.py
Author: Koldo Pina
Date created: 04/02/2018
Date last modified: 04/02/2018
Python Version: 3.5
"""

# Agrupando con Pandas
import pandas as pd

df = pd.read_csv('trabajos_02.csv', sep=";", decimal=",")
grouped_df = df.groupby(['ZONA', 'PLANTA', 'DIA_MES']).sum().reset_index()
print(grouped_df)
