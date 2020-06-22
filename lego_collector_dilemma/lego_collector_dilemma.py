# File path of lego_final.csv stored in path

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
df = pd.read_csv(path)
print(df.head())

X = df.drop('list_price',axis=1)
y = df.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)
print(X_train.shape)

import matplotlib.pyplot as plt
cols = X_train.columns
print(cols.shape)
fig ,axes = plt.subplots(nrows=3, ncols=3, figsize=(15,10))
for i in range(0,3) :
    for j in range(0,3) :
        col = cols[i*3+j]
        axes[i,j].scatter(X_train[col], y_train)
plt.show()

corr = X_train.corr()
X_train.drop(['play_star_rating','val_star_rating'], inplace=True, axis=1)
X_test.drop(['play_star_rating','val_star_rating'], inplace=True, axis=1)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(mse)
r2 = r2_score(y_test, y_pred)
print(r2)

residual = y_test - y_pred

