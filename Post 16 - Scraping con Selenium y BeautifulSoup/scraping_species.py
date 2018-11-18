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

species_urls = ['https://en.wikipedia.org/wiki/List_of_Star_Wars_species_(A%E2%80%93E)',
                'https://en.wikipedia.org/wiki/List_of_Star_Wars_species_(F%E2%80%93J)',
                'https://en.wikipedia.org/wiki/List_of_Star_Wars_species_(K%E2%80%93O)',
                'https://en.wikipedia.org/wiki/List_of_Star_Wars_species_(P%E2%80%93T)',
                'https://en.wikipedia.org/wiki/List_of_Star_Wars_species_(U%E2%80%93Z)']

start_time = datetime.now()
print (f'Start time: {start_time}')

def parse_species(url):
    species_list = []
    url_species_req = requests.get(url)
    species_html = url_species_req.text
    species_soup = BeautifulSoup(species_html, "html.parser")

    species = species_soup.find_all("span", {'class': 'toctext'})
    for specie in species:
        name = specie.get_text().strip()
        if name not in ('References', 'External links', 'Bibliography'):
            species_list.append([name])
    return species_list

starwars_species_list = []

for url in species_urls:
    species_sublist = parse_species(url)
    starwars_species_list += species_sublist

starwars_species_df = pd.DataFrame(starwars_species_list, columns = ['name'])
now = datetime.now()
filename  = f'starwars_species_{now.year}{now.month}{now.day}.csv'
starwars_species_df.to_csv(filename, sep = ';', index = False, encoding = 'utf-8')

end_time = datetime.now()
print (f'End time: {end_time}')
total_time = end_time - start_time
print (f'Finished in: {total_time}')
#Finished in: 0:00:02.234633
