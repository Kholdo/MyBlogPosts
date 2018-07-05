#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Koldo Pina
Date created: 05/07/2018
Python Version: 3.6
"""

import pandas as pd

df_adaline = pd.read_csv('adaline_data.csv', sep=';')
df_perceptron = pd.read_csv('perceptron_data.csv', sep=';')

def describe_plus(df):
    """
    Function that returns describe dataframe plus number of NaNs, number of uniques values, the mode
    # and the sts per column
    :param df: dataframe
    :return: dataframe
    """
    describe = df.describe()
    describe.loc['NaNs'] = [df_adaline[feature].isnull().sum() for feature in df.columns]
    describe.loc['uniques'] = [len(df_adaline[feature].unique()) for feature in df.columns]
    describe.loc['mode'] = [df_adaline[feature].mode()[0] for feature in df.columns]
    describe.loc['std'] = [df_adaline[feature].std() for feature in df.columns]
    return describe.transpose()

print (describe_plus(df_adaline))

print (describe_plus(df_perceptron))