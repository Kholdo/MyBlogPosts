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

fig = plt.figure()
sg = fig.add_subplot(111)

sg.set_title('función sigmoide',fontsize=16, fontweight='bold')

sg.set_xlabel('x')
sg.set_ylabel('y')

sg.text(-7, 0.9, r'$y=\frac{1}{1+e^{-f(x)}}$', style='italic',fontsize=15,
        bbox={'facecolor': 'green', 'alpha': 0.3, 'pad': 10})

sg.text(-15, 0.53, r'pto inflexión: y=0.5', fontsize=15, color='red')

plt.plot(x, y, color='#1F5320', lw = 3.0)
plt.plot(x, y_mid, color='red', lw = 2.0, ls = 'dashed')

plt.show()