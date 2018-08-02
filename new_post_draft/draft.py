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

    def mv(self, yi, pi):
        """
        funcion para el calculo de la funcion de maxima verosimilitud
        :param yi: lista de las variables respuesta
        :param pi: lista de las probabilidades asociadas a cada suceso
        :return: calculo de la funcion de maxima verosimilitud
        """
        import numpy as np

        L = np.prod([float(np.where(y==1, pi[i], 1-pi[i])) for i, y in enumerate(yi)])

        return L

    def sig(self, X, B):
        """
        función para el calculo de la función sigmoide
        :param X: Matriz de las variables predictoras
        :param B: Matriz de los coeficientes
        :return: Matriz de las probabilidades asociadas a cada suceso
        """
        import numpy as np

        exponents = []
        pi = []

        for i, row in enumerate(X):
            exponent = sum([x * B[index] for index, x in enumerate(row)])
            exponents.append(exponent)
            try:
                p = 1 / (1 + np.exp(-exponent))
            except:
                p = np.nan
            pi.append(p)

        return pi