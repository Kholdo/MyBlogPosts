#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 18/02/2018
Python Version: 3.5
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

lego_stores_list = []

lego_stores_url = 'https://www.lego.com/en-us/stores/stores'
lego_stores_req = requests.get(lego_stores_url)
lego_stores_html = lego_stores_req.text
lego_stores_soup = BeautifulSoup(lego_stores_html, "html.parser")

div = lego_stores_soup.find_all('div', {"class": "row-block mark-box"})
for div_block in div:
    h3 = div_block.find('h3')
    lego_store_country = h3.get_text().strip()
    print (lego_store_country)
    lego_stores_links = div_block.find_all("a", {"class": "btn-details"}, href=True)
    for link in lego_stores_links:
        print(link.get('href'))
        store_ulr = link.get('href')
        store_req = requests.get(store_ulr)
        store_soup = BeautifulSoup(store_req.text, "html.parser")
        store_address = store_soup.find("address")
        link_address = store_address.contents[0].strip() + ' ' + store_address.contents[2].strip()
        if lego_store_country == 'China':
            link_address = link_address.encode('utf8')
        lego_stores_list.append([lego_store_country, store_ulr, link_address])

lego_stores_df = pd.DataFrame(lego_stores_list, columns=['country', 'store_ulr', 'address'])
now = datetime.now()
filename = 'Py_Lego_stores_%d%d%d.csv' % (now.year, now.month, now.day)
lego_stores_df.to_csv(filename, sep=';', index=False, encoding='utf-8')