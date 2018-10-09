#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 23/09/2018
Python Version: 3.6
"""

import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
from datetime import datetime

start_time = datetime.now()
print (f'Start time: {start_time}')

driver = webdriver.Firefox()
url = 'http://www.dimfuture.net/starwars/random/generate.php'
driver.get(url)

names = []
while len(names)<100000:
    try:
        #Click on Radio button with value 100
        one_hundred = driver.find_element_by_xpath("//input[@name='choice' and @value='100']")
        one_hundred.click()

        #Click on Generate! button
        generate = driver.find_element_by_xpath("//input[@name='submit' and @value='Generate!']")
        generate.click()

        #Transfer info to BeautifulSoup
        starwars_names_soup = BeautifulSoup(driver.page_source, 'lxml')
        table = starwars_names_soup.find_all('table')
        rows = table[3].find_all("td")
        for row in rows:
            newname = row.get_text().strip().replace(u'\xa0', u' ')
            if newname not in names:
                names.append([newname])
    except:
        pass
    print(len(names))

driver.close()

starwars_names_df = pd.DataFrame(names, columns = ['name'])
now = datetime.now()
filename  = f'starwars_names_{now.year}{now.month}{now.day}.csv'
starwars_names_df.to_csv(filename, sep = ';', index = False, encoding = 'utf-8')

end_time = datetime.now()
print (f'End time: {end_time}')
total_time = end_time - start_time
print (f'Finished in: {total_time}')
#Finished in: 1:43:59.618806