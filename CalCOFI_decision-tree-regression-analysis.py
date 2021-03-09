#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The CalCOFI data: Decision Tree Regression Analysis
(California Cooperative Oceanic Fisheries Investigations)

@author: richasharma
"""


# Importing the main libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset 
dataset1 = pd.read_csv('bottle.csv', encoding='latin_1')

print(dataset1.head())


# We only need two columns to understand the 
#relation between temeperature and water salinity

botle_df = dataset1[['Salnty','T_degC']]
print(botle_df.head())


# Data Pre-processing

print(botle_df.isnull().sum()) # print null values

botle_df.fillna(method='ffill', inplace=True)

print(botle_df.isnull().sum())


# We take only few data for regression for quick calculations

botle_df = botle_df[:][:500]  


X = botle_df.iloc[:, :-1].values
y = botle_df.iloc[:, -1].values


# Splitting the data into training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 1)


# Training the Decision Tree Regression model on the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)



# Predicting result

y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))


#0.849108143486555













