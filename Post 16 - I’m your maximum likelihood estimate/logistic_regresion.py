#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 23/09/2018
Python Version: 3.6
"""

#Import the modules we are going to need
import pandas as pd
import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

#Step one: Create a dataset

def genDataset(n_features=4, n_samples=10000, low=0, high=10, weights=[0.25, 0.25, 0.25, 0.25], threshold=5):
    """
    The function generates a random dataset with n predictor features and the result. The result
    can take two values, zero and one. The result depends on the threshold.

    :param n_features: Number of features.
    :param n_samples: Number of samples.
    :param low: The lowest value for the range of random values.
    :param high: The highest value for the range of random values.
    :param weight: List of the weights for each feature. There must be a weight for each feature.
    :param threshold: It is the value that marks the limit to assign zero or one to the result.

    :return: Dataframe

    """
    import pandas as pd
    import numpy as np

    if len(weights) == n_features:
        randomData = np.random.randint(low=low, high=high, size=(n_samples, n_features))
        weights_m = np.asarray([weights] * n_samples)
        p = np.array([1 if sum(row) >= threshold else 0 for row in randomData * weights_m])
        res = pd.DataFrame(data=np.column_stack((randomData, p)),
                           columns=['feat_%s_%s' % (str(i), str(w)) for i, w in enumerate(weights)] + [
                               'result_thr_%s' % threshold])
        if sum(weights) != 1.0:
            print("Beware. The sum of the weights is different from 1.0: %f" % sum(weights))
        return res
    else:
        print("There must be a weight for each feature. Please, check the weights matrix.")

data = genDataset(n_features = 4, n_samples = 50000, low = 0, high = 10,
                  weights =  [0.125, 0.375, 0.125, 0.375],
                  threshold = 6)

data.columns = ['f1', 'f2', 'f3', 'f4', 'y']

#Step two: Data exploration
print (data.describe().transpose())

fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharey=True)
pd.crosstab(data['f1'],data['y']).plot( kind='bar', ax = axs[0][0], title = 'Values Frequency for f1')
pd.crosstab(data['f2'],data['y']).plot(kind='bar', ax = axs[0][1], title = 'Values Frequency for f2')
pd.crosstab(data['f3'],data['y']).plot(kind='bar', ax = axs[1][0], title = 'Values Frequency for f3')
pd.crosstab(data['f4'],data['y']).plot(kind='bar', ax = axs[1][1], title = 'Values Frequency for f4')
plt.show()

sns.countplot(x='y', data=data, palette='hls')
plt.show()

count_zeros = len(data[data['y']==0])
count_ones = len(data[data['y']==1])
percent_zeros = count_zeros/(count_zeros+count_ones)
print(f'zeros percentage: {percent_zeros*100}')
percent_ones = count_ones/(count_zeros+count_ones)
print(f'ones percentage: {percent_ones*100}')


#Step three: Create logistic regression model

class LogReg_mv():

    def __init__(self, increment, X, Y):
        """
        :param increment: is the limit increase for the algorithm to converge
        :param X: matrix of the predictor variables
        :param Y: matrix of the response variables
        """
        self.increment = increment
        self.X = X
        self.Y = Y
        # We give an initial value of 0 to the coefficients. Matriz: one column, n rows.
        self.B = np.zeros(X.shape[1]).reshape(X.shape[1], 1)

    def w_matrix(self, pi):
        """
        function for calculating the W matrix
        :param pi: matrix of probabilities associated with each event
        :return: matrix W
        """
        w = np.identity(len(pi))
        for index, p in enumerate(pi):
            w[index][index] = float(p * (1 - p))
        return w

    def prob_matrix(self, B):
        """
        function for the calculation of the sigmoid function
        :param B: matrix of the coefficients
        :return: matrix of probabilities associated with each event
        """
        import numpy as np

        exponents = []
        pi = []

        for i, row in enumerate(self.X):
            exponent = sum([x * B[index] for index, x in enumerate(row)])
            exponents.append(exponent)
            p = 1 / (1 + np.exp(-exponent))
            pi.append(p)

        return pi

    def fit(self):
        """
        function that calculates the coefficients of the logistic regression
        """
        import numpy as np
        from numpy import linalg

        self.num_iter = 0
        self.increments_list = []

        while True:
            p = self.prob_matrix(self.B)
            w = self.w_matrix(p)

            first_order_der = np.dot(np.transpose(self.X), np.subtract(self.Y, np.transpose(p)).transpose())
            second_order_der = np.transpose(self.X).dot(w).dot(self.X)
            # delta: incremento
            delta = np.transpose(np.dot(np.transpose(first_order_der), linalg.inv(second_order_der)))

            self.B = self.B + delta

            self.num_iter = self.num_iter + 1

            increment = np.sum(np.power(delta, 2))
            self.increments_list.append(increment)
            if increment <= self.increment:
                break

        equation = f'Y = '
        for i, b in enumerate(self.B):
            if i == 0:
                equation_2 = f'{round(b[0],4)} '
            else:
                if b[0] < 0:
                    equation_2 = equation_2 + f'{round(b[0],4)}⋅X{i} '
                else:
                    equation_2 = equation_2 + f'+ {round(b[0],4)}⋅X{i} '

        print('Logistic regresion model fitted')
        print(f'Completed in {self.num_iter} iterations')
        print('Increments')
        print(self.increments_list)
        print('Coefficients:')
        print(self.B)
        print('Equation:')
        print(equation + equation_2)

        # return self.B

    def predict(self, X):
        """
        function that calculates the values for the response variable
        :param X: test dataset with the predictors variables
        :return p_list: list with the values for the response variable.
        """
        p_list = []
        for ele in np.array(X):
            y = float(np.dot(ele, self.B))
            p = 1 / (1 + np.exp(-y))
            if p >= 0.5:
                p_list.append(1)
            else:
                p_list.append(0)
        return p_list