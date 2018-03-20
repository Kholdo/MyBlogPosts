#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 18/03/2018
Python Version: 3.5
"""

import pandas as pd

#Importando los datos
df = pd.read_csv('data_LR.dsv', sep="|")
df = df.replace(',','.', regex=True).astype(float)

