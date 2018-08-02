#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 02/08/2018
Python Version: 3.6
"""

class LogReg():

    def __init__(self, error, n_iter):
        """
        :param error: error gradient descent
        :param n_iter: número de iteraciones(epochs)
        """
        self.error = eta
        self.n_iter = n_iter

    def LH(self, yi, pi):
        """
        :param yi: lista de las variables respuesta
        :param pi: lista de las probabilidades asociadas a cada suceso
        :return: cálculo de la función de máxima verosimilitud
        """
        import numpy as np
        L = np.prod([float(np.where(y==1, pi[i], 1-pi[i])) for i, y in enumerate(yi)])
        return L
