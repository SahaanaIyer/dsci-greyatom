# File path of cleaned_loan_data.csv stored in path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(path)

# Independence Check
fico_gt = len(df[df['fico'] > 700])
fico_total = len(df['fico'])
p_a = fico_gt / fico_total
purpose_gt = len(df[df['purpose'] == 'debt_consolidation'])
purpose_total = len(df['purpose'])
p_b = purpose_gt / purpose_total
df1 = df[df['purpose']=='debt_consolidation']
p_a_int_b = len(df[(df['fico']>700) & (df['purpose']=='consolidation')])
p_a_b = p_a_int_b / p_a
p_b_a = p_a_int_b / p_b
if p_b_a == p_a :
    result = 1
else :
    result = 0
print(result)

# Bayes theorem
pbl_yes = len(df[df['paid.back.loan']=='Yes'])
pbl = len(df['paid.back.loan'])
prob_lp = pbl_yes / pbl

cp_yes = len(df[df['credit.policy']=='Yes'])
cp = len(df['credit.policy'])
prob_cs = cp_yes / cp

new_df = df[df['paid.back.loan']=='Yes']
prob_pd_cs = new_df[new_df['credit.policy'] == 'Yes'].shape[0] / new_df.shape[0]

print(prob_pd_cs)
bayes = (prob_pd_cs * prob_lp)/ prob_cs
print(bayes)

# Purpose vs paid back loan
df.purpose.value_counts(normalize=True).plot(kind='bar')
df1 = df[df['paid.back.loan']=='No']
df1.purpose.value_counts(normalize=True).plot(kind='bar')
plt.show()

# Visualization of continuous data
inst_median = df['installment'].median()
inst_mean = df['installment'].mean()
df.hist('installment', bins=10)
df.hist('log.annual.inc', bins=10)

