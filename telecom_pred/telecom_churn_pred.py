# File path of telecom_churn.csv is stored in path

import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv(path)
X = df.iloc[:,1:-1]
y = df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

import numpy as np
from sklearn.preprocessing import LabelEncoder
X_train['TotalCharges'] = X_train['TotalCharges'].replace(" ",np.NaN)
X_test['TotalCharges'] = X_test['TotalCharges'].replace(" ",np.NaN)
X_train['TotalCharges'] = X_train['TotalCharges'].astype(float)
X_test['TotalCharges'] = X_test['TotalCharges'].astype(float)
X_train['TotalCharges'] = X_train['TotalCharges'].fillna(X_train['TotalCharges'].mean())
X_test['TotalCharges'] = X_test['TotalCharges'].fillna(X_test['TotalCharges'].mean())
print(X_train['TotalCharges'].isnull().sum())
print(X_test['TotalCharges'].isnull().sum())
le = LabelEncoder()
X_train_feature_mask = X_train.dtypes==object
X_train_cols = X_train.columns[X_train_feature_mask].tolist()
X_train[X_train_cols] = X_train[X_train_cols].apply(lambda col: le.fit_transform(col))
X_test_feature_mask = X_test.dtypes==object
X_test_cols = X_test.columns[X_test_feature_mask].tolist()
X_test[X_test_cols] = X_test[X_test_cols].apply(lambda col: le.fit_transform(col))
y_train = y_train.replace({'No':0, 'Yes':1})
y_test = y_test.replace({'No':0, 'Yes':1})

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
ada_model = AdaBoostClassifier(random_state=0)
ada_model.fit(X_train, y_train)
y_pred = ada_model.predict(X_test)
ada_score = accuracy_score(y_test, y_pred)
ada_cm = confusion_matrix(y_test, y_pred)
ada_cr = classification_report(y_test, y_pred)

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}
xgb_model = XGBClassifier(random_state=0)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
xgb_score = accuracy_score(y_test, y_pred)
xgb_cm = confusion_matrix(y_test, y_pred)
xgb_cr = classification_report(y_test, y_pred)
print('{}\t{}\t{}'.format(xgb_score, xgb_cm, xgb_cr))
clf_model = GridSearchCV(estimator=xgb_model, param_grid=parameters)
clf_model.fit(X_train, y_train)
y_pred = clf_model.predict(X_test)
clf_score = accuracy_score(y_test, y_pred)
clf_cm = confusion_matrix(y_test, y_pred)
clf_cr = classification_report(y_test, y_pred)
print('{}\t{}\t{}'.format(clf_score, clf_cm, clf_cr))

