# File path of googleplaystore.csv is stored in path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv(path)
data.hist('Rating', bins=10)
data = data[data['Rating']<=5.0]
data.hist('Rating', bins=10)

total_null = data.isnull().sum()
percent_null = total_null/data.isnull().count()
missing_data = pd.concat([total_null, percent_null], keys=['Total','Percent'], axis=1)
print(missing_data)
data.dropna(inplace=True)
total_null_1 = data.isnull().sum()
percent_null_1 = (total_null_1/data.isnull().count())
missing_data_1 = pd.concat([total_null_1, percent_null_1], axis=1, keys=['Total', 'Percent'])
print(missing_data_1)

sns.catplot(x="Category", y="Rating", data=data, kind="box", height = 10)
plt.xticks(rotation=90)
plt.title("Rating vs Category [BoxPlot]")
plt.show()

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import re
print(data['Installs'].value_counts())
data['Installs'] = data['Installs'].str.replace('+', ' ', regex=True)
data['Installs'] = data['Installs'].str.replace(',', '', regex=True)
data['Installs'] = data['Installs'].astype(int)
print(data['Installs'])
le = LabelEncoder()
data['Installs'] = le.fit_transform(data[['Installs']])
sns.regplot(x="Installs", y="Rating", data=data)
plt.title("Rating vs Installs [RegPlot]")

print(data['Price'].value_counts())
data['Price'] = data['Price'].str.replace('$', ' ', regex=True)
data['Price'] = data['Price'].astype(float)
print(data['Price'])
le = LabelEncoder()
data['Price'] = le.fit_transform(data[['Price']])
sns.regplot(x="Price", y="Rating", data=data)
plt.title("Rating vs Price [RegPlot]")

print(data['Genres'].unique())
data['Genres'] = data['Genres'].str.split(";").str[0]
print(data['Genres'])
gr_mean = data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean().sort_values('Rating')

data['Last Updated'] = pd.to_datetime(data['Last Updated'])
data['Last Updated Days'] = (data['Last Updated'].max()-data['Last Updated'] ).dt.days
plt.figure(figsize = (10,10))
sns.regplot(x="Last Updated Days", y="Rating", color = 'lightpink',data=data )
plt.title('Rating vs Last Updated [RegPlot]',size = 20)
