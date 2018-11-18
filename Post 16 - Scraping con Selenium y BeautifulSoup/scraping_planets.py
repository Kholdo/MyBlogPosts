#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 30/09/2018
Python Version: 3.6
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

start_time = datetime.now()
print (f'Start time: {start_time}')

starwars_planet_list= []

url_planets = 'http://starwars.wikia.com/wiki/List_of_planets'

url_planets_req = requests.get(url_planets)
planets_html = url_planets_req.text
planets_soup = BeautifulSoup(planets_html, "html.parser")

tables = planets_soup.find_all("tr")
for t, table in enumerate(tables):
    for i, value in enumerate(table.find_all("td")):
        if i == 0 and t >= 6:
            starwars_planet_list.append([value.get_text().strip()])

starwars_planets_df = pd.DataFrame(starwars_planet_list, columns = ['name'])
now = datetime.now()
filename  = f'starwars_planets_{now.year}{now.month}{now.day}.csv'
starwars_planets_df.to_csv(filename, sep = ';', index = False, encoding = 'utf-8')

end_time = datetime.now()
print (f'End time: {end_time}')
total_time = end_time - start_time
print (f'Finished in: {total_time}')
#Finished in: 0:00:00.437466


