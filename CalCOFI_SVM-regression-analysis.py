#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The CalCOFI data: Support Vector Regression Analysis
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
y = y.reshape(len(y),1)   # As transformation expects a 2-D Array



# Splitting the data into training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 1)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(y_train)
print(X_train)
print(y_train)



# Training the SVR model on the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)



# Predicting a new result

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))


from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

#0.9222822313319194














