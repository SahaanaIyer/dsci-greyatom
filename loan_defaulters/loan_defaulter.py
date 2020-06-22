# File path of cleaned_loan_data.csv is stored in path

import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv(path)
y = data['paid.back.loan']
X = data.iloc[:, 1:-1]
print(X.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=0)

import matplotlib.pyplot as plt
fully_paid = y_train.value_counts()
plt.bar(fully_paid.index, fully_paid)

import numpy as np
from sklearn.preprocessing import LabelEncoder
X_train['int.rate'] = X_train['int.rate'].str.replace('%', ' ')
X_train['int.rate'] = (X_train['int.rate'].astype(float)) / 100
X_test['int.rate'] = X_test['int.rate'].str.replace('%', ' ')
X_test['int.rate'] = (X_test['int.rate'].astype(float)) / 100
num_df = X_train.select_dtypes(include=['number'])
cat_df = X_test.select_dtypes(include=['object'])
print(cat_df)

import seaborn as sns
cols = num_df.columns
fig ,axes = plt.subplots(nrows=9, ncols=1, figsize=(15,10))
for i in range(0,9) :
    sns.boxplot(x=y_train, y=num_df[cols[i]], ax=axes[i])

cols = cat_df.columns
fig ,axes = plt.subplots(nrows=2, ncols=2)
for i in range(0,2) :
    for j in range(0,2) :
        sns.countplot(x=X_train[cols[i*2+j]], hue=y_train, ax=axes[i,j])

from sklearn.tree import DecisionTreeClassifier
for i in cat_df.columns :
    X_train[i].fillna('NA', inplace=True)
    le = LabelEncoder()
    X_train[i] = le.fit_transform(X_train[i])
    X_test[i].fillna('NA', inplace=True)
    X_test[i] = le.transform(X_test[i])
y_train.replace({'No': 0, 'Yes': 1}, inplace=True)
y_test.replace({'No': 0, 'Yes': 1}, inplace=True)
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)

from sklearn.model_selection import GridSearchCV
parameter_grid = {'max_depth': np.arange(3,10), 'min_samples_leaf': range(10,50,10)}
model_2 = DecisionTreeClassifier(random_state=0)
p_tree = GridSearchCV(estimator=model_2, param_grid=parameter_grid, cv=5)
p_tree.fit(X_train, y_train)
acc_2 = p_tree.score(X_test, y_test)

from io import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn import metrics
from IPython.display import Image
import pydotplus
dot_data = export_graphviz(decision_tree=p_tree.best_estimator_, out_file=None, feature_names=X.columns, filled = True, class_names=['loan_paid_back_yes','loan_paid_back_no'])
graph_big = pydotplus.graph_from_dot_data(dot_data)
img_path = user_data_dir+'/file.png'
graph_big.write_png(img_path)
plt.figure(figsize=(20,15))
plt.imshow(plt.imread(img_path))
plt.axis('off')
plt.show()
