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