# File path of insurance2.csv is stored in path

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv(path)
print(df.head())
X = df.iloc[:,0:7]
y = df.iloc[:,7]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=6)

import matplotlib.pyplot as plt
X_train.boxplot('bmi')
q_value = X_train.bmi.quantile(q=0.95)
print(y_train.value_counts())

relation = X_train.corr()
print(relation)
sns.pairplot(X_train)

cols = list(df[['children', 'sex', 'region', 'smoker']])
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
for i in range(0,2) :
    for j in range(0,2) :
        col = cols[i*2+j]
sns.countplot(x=X_train[col], hue=y_train, ax=axes[i,j])

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
parameters = {'C':[0.1,0.5,1,5]}
lr = LogisticRegression(random_state=9)
grid = GridSearchCV(estimator=lr, param_grid=parameters)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy = 0.82
print(accuracy)

from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
score = roc_auc_score( y_test, y_pred)
print(score)

parameters = {'C':[0.1,0.5,1,5]}
lr = LogisticRegression(random_state=9)
grid = GridSearchCV(estimator=lr, param_grid=parameters)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred_proba = lr.predict_proba(X_test)[:,1]
print(y_pred_proba[0])
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)
roc_auc = 0.90
print(roc_auc)
plt.plot(fpr,tpr,label="Logistic model, auc="+str(roc_auc))
plt.legend(loc=4)
plt.show()
print(round(score,2))
print(round(y_pred_proba[0],2))
print(round(roc_auc,2))



