#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 26/07/2018
Python Version: 3.6
"""

import numpy
import matplotlib.pyplot as plt

def sigmoide(x):
    res = [1 / (1 + np.exp(-ele)) for ele in x]
    return res

x = np.arange(-15, 15, 0.5)
y = sigmoide(x)
y_mid = [0.5 for ele in x]

