#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 26/07/2018
Python Version: 3.6
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoide(x):
    res = [1 / (1 + np.exp(-ele)) for ele in x]
    return res

x = np.arange(-15, 15, 0.5)
y = sigmoide(x)
y_mid = [0.5 for ele in x]

fig = plt.figure(figsize=(12,8))
plt.plot(x, y, color='#1F5320', lw = 3.0)
plt.plot(x, y_mid, color='red', lw = 2.0, ls = 'dashed')
plt.ylabel('Y')
plt.xlabel('X')
plt.suptitle('función sigmoide: ' + r'$y=\frac{1}{1+e^{-f(x)}}$', size = 20)
plt.grid(True)
plt.show()