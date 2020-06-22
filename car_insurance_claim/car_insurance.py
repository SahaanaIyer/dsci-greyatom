# File path of car_insurance_claim.csv is stored in path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.read_csv(path)
print(df.head())
print(df.info)
for col in ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT'] :
     df[col].replace({"\$": "", ",":""}, regex=True, inplace=True)
y = df['CLAIM_FLAG']
X = df.iloc[:,0:-1]
count = y.value_counts()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 6)

X_train[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']] = X_train[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].astype(float)
X_test[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']] = X_test[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].astype(float)
print(X_train.isnull().sum())
print(X_test.isnull().sum())

X_train.dropna(subset=['YOJ','OCCUPATION'], inplace=True)
X_test.dropna(subset=['YOJ','OCCUPATION'], inplace=True)
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]
X_train[["AGE","CAR_AGE","INCOME","HOME_VAL"]].fillna(X_train[["AGE","CAR_AGE","INCOME","HOME_VAL"]].mean(), inplace=True)
X_test[["AGE","CAR_AGE","INCOME","HOME_VAL"]].fillna(X_test[["AGE","CAR_AGE","INCOME","HOME_VAL"]].mean(), inplace=True)

from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1", "MSTATUS", "GENDER", "EDUCATION", "OCCUPATION", "CAR_USE", "CAR_TYPE", "RED_CAR", "REVOKED"]
for col in columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col]).astype(str)
    X_test[col] = le.transform(X_test[col]).astype(str)

from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 6)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=9)
X_train, y_train = smote.fit_sample(X_train, y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)