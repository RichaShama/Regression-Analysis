#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The CalCOFI data: Polynomial Linear Regression Analysis
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
print(dataset2.head())

savename = "TempVsSalinity-polynomial"

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


# Training the Linear Regression model on the whole dataset

X = botle_df.iloc[:, :-1].values
y = botle_df.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# Training the Polynomial Regression model on the whole dataset

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
poly_predict = lin_reg_2.predict(X_poly)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'green', s=14)
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.ylabel(r'Temperature ($^{\degree}$ Celsius)')
plt.xlabel('Salinity')
plt.show()



# Visualising the Polynomial Regression results (for higher resolution and smoother curve)

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'green', s=14)
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.ylabel(r'Temperature ($^{\degree}$ Celsius)')
plt.xlabel('Salinity')
plt.savefig(savename+'.pdf', dpi=600, bbox_inches='tight')
plt.show()


from sklearn.metrics import r2_score

print(r2_score(y,poly_predict))


#0.9196159256964279
















