#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 27/06/2018
Python Version: 3.5
"""


def genDataset(n_features=4, n_samples=1000, low=0, high=10, weights=[0.25, 0.25, 0.25, 0.25], threshold=5):
    """
    The function generate a random dataset with n predictor features and the result. The result
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