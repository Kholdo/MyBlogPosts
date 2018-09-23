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