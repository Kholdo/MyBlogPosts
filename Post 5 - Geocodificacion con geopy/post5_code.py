#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 18/02/2018
Python Version: 3.5
"""

import pandas as pd
from geopy.geocoders import GoogleV3

df = pd.read_csv('Py_Lego_stores_2018218.csv', sep=";")


def geolego(row):
    address_src = row['address']
    print(row['country'])
    if row['country'] == 'China':
        address_src = eval(address_src).decode('utf-8')
    try:
        address, (latitude, longitude) = geolocator.geocode(address_src, timeout=15)
    except TypeError:
        address, latitude, longitude = 'TypeError', -91, -181
    print(address, latitude, longitude)
    return pd.Series({"GOOGLE_address": address, "latitude": latitude, "longitude": longitude})


df[['GOOGLE_address', 'latitude', 'longitude']] = df.apply(geolego, axis=1)

df = df[df.GOOGLE_address != 'TypeError']
print(df.head())

df.to_csv('lego_stores_coord.csv', sep=';', index=False, encoding='utf-8')
