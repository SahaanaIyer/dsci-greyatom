# File path of train.csv is stored in path

import pandas as pd
from sklearn import preprocessing
dataset = pd.read_csv(path)
print(dataset.head())
dataset.drop('Id', inplace=True, axis=1)

import seaborn as sns
from matplotlib import pyplot as plt
cols = dataset.columns
size = len(cols)-1
x = cols[size]
y = cols[0:size]
for i in range(0,size):
    sns.violinplot(data=dataset,x=x,y=y[i])
    plt.show()

upper_threshold = 0.5
lower_threshold = -0.5
subset_train = dataset.iloc[:, :10]
data_corr = subset_train.corr()
f, ax = plt.subplots(figsize = (10,8))
sns.heatmap(data_corr,vmax=0.8,square=True);
correlation = data_corr.unstack().sort_values(kind='quicksort')
corr_var_list = correlation[((correlation>upper_threshold) | (correlation<lower_threshold)) & (correlation!=1)]
print(corr_var_list)

from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)
r,c = dataset.shape
X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)
scaler = StandardScaler()
X_train_temp = scaler.fit_transform(X_train.iloc[:,:10])
X_test_temp = scaler.transform(X_test.iloc[:,:10])
X_train1 = numpy.concatenate((X_train_temp,X_train.iloc[:,10:c-1]),axis=1)
X_test1 = numpy.concatenate((X_test_temp,X_test.iloc[:,10:c-1]),axis=1)
scaled_features_train_df = pd.DataFrame(X_train1, index=X_train.index, columns=X_train.columns)
scaled_features_test_df = pd.DataFrame(X_test1, index=X_test.index, columns=X_test.columns)

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
skb = SelectPercentile(score_func=f_classif,percentile=90)
predictors = skb.fit_transform(X_train1, Y_train)
scores = list(skb.scores_)
Features = scaled_features_train_df.columns
dataframe = pd.DataFrame({'Features':Features,'Scores':scores})
dataframe=dataframe.sort_values(by='Scores',ascending=False)
top_k_predictors = list(dataframe['Features'][:predictors.shape[1]])
print(top_k_predictors)

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
lr = LogisticRegression()
clf = OneVsRestClassifier(lr)
clf1 = OneVsRestClassifier(lr)
model_fit_all_features = clf1.fit(X_train, Y_train)
predictions_all_features = clf1.predict(X_test)
score_all_features = accuracy_score(Y_test, predictions_all_features)
model_fit_top_features = clf.fit(scaled_features_train_df[top_k_predictors], Y_train)
predictions_top_features = clf.predict(scaled_features_test_df[top_k_predictors])
score_top_features = accuracy_score(Y_test, predictions_top_features)