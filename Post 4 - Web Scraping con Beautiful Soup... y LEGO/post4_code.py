#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 08/02/2018
Python Version: 3.5
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime

lego_stores_list = []
lego_stores_url = 'https://www.lego.com/en-us/stores/stores'
lego_stores_req = requests.get(lego_stores_url)

