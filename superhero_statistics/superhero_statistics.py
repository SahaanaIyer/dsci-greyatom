# File path of superhero_data.csv is stored in path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data loading
data = pd.read_csv(path)
data['Gender'].replace('-','Agender',inplace=True)
gender_count = data['Gender'].value_counts()
print(gender_count)
plt.bar(gender_count.index, gender_count)
plt.show()

# Heros Alignment
alignment = data['Alignment'].value_counts()
plt.pie(alignment)
plt.show()

# Combat Correlation
sc_df = data[['Strength','Combat']]
sc_covariance = sc_df.cov().iloc[0,1]
sc_strength = sc_df.Strength.std()
print(sc_strength)
sc_combat = sc_df.Combat.std()
print(sc_combat)
sc_pearson = sc_covariance / (sc_strength * sc_combat)
print(sc_pearson)

ic_df = data[['Intelligence','Combat']]
ic_covariance = ic_df.cov().iloc[0,1]
ic_intelligence = ic_df.Intelligence.std()
print(ic_intelligence)
ic_combat = ic_df.Combat.std()
print(ic_combat)
ic_pearson = ic_covariance / (ic_intelligence * ic_combat)
print(ic_pearson)

# Overpowered Super Beings
total_high = data.Total.quantile(0.99)
super_best = data[data['Total'] > total_high]
super_best_names = list(super_best['Name'])
print(super_best_names)

# Statistics for Survival
fig, (ax_1, ax_2, ax_3) = plt.subplots(1,3, figsize=(20,10))
ax_1.boxplot(super_best['Intelligence'])
ax_1.set_title('Intelligence')
ax_2.boxplot(super_best['Speed'])
ax_2.set_title('Speed')
ax_1.boxplot(super_best['Power'])
ax_3.set_title('Power')
plt.show()