# File path of cleaned_loan_data.csv stored in path

import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
sample_size=2000
z_critical = stats.norm.ppf(q = 0.95)
data = pd.read_csv(path)
data_sample = data.sample(n=sample_size, random_state=0)
sample_mean = data_sample['installment'].mean()
sample_std = data_sample['installment'].std()
margin_of_error = (z_critical * sample_std) / math.sqrt(sample_size)
confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
true_mean = data['installment'].mean()
print(margin_of_error)
print(true_mean)
print(confidence_interval[0])
print(confidence_interval[1])

sample_size = np.array([20, 50, 100])

fig, (ax_1, ax_2, ax_3) = plt.subplots(nrows=3, ncols=1, figsize=(20, 10))
for i in range(len(sample_size)):
    m = []
    for j in range(1000):
        m.append(data['installment'].sample(n=sample_size[i]).mean())
    mean_series = pd.Series(m)

from statsmodels.stats.weightstats import ztest
data['int.rate'] = data['int.rate'].map(lambda x : str(x).replace('%', ''))
data['int.rate'] = data['int.rate'].astype(float).div(100)
z_statistic, p_value = ztest(x1 = data[data['purpose']=='small_business']['int.rate'],
                             value = data['int.rate'].mean(),
                             alternative = 'larger')
print(z_statistic)
print(p_value)
if p_value>=0.05 :
    inference = 'Accept'
else :
    inference = 'Reject'
print(inference)

z_statistic, p_value = ztest(x1 = data[data['paid.back.loan']=='No']['installment'],
                             x2 = data[data['paid.back.loan']=='Yes']['installment'] )
if p_value >= 0.05 :
    inference = 'Accept'
else :
    inference = 'Reject'
print(inference)

from scipy.stats import chi2_contingency

#Critical value
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

yes = data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
no = data[data['paid.back.loan']=='No']['purpose'].value_counts()
observed = pd.concat([yes.transpose(), no.transpose()], 1, keys=['Yes', 'No'])
chi2, p, dof, ex = chi2_contingency(observed, correction=False)
if chi2 > critical_value :
    inference = 'Accept'
else :
    inference = 'Reject'

