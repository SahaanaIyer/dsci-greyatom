# File path of loan_prediction.csv is stored in path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv(path)
loan_status = data['Loan_Status'].value_counts()
plt.bar(loan_status.index, loan_status)
plt.show()

property_and_loan = data.groupby(['Property_Area','Loan_Status'])
property_and_loan = property_and_loan.size().unstack()
property_and_loan.plot(kind='bar', stacked=False, figsize=(15,10))
plt.xlabel('Property Area')
plt.ylabel('Loan Status')
plt.xticks(rotation=45)
plt.show()

education_and_loan = data.groupby(['Education','Loan_Status']).size().unstack()
education_and_loan.plot(kind="bar", stacked=True, figsize=(15,10))
plt.xlabel('Education Status')
plt.ylabel('Loan Status')
plt.xticks(rotation=45)
plt.show()

graduate = data[data["Education"]=='Graduate']
not_graduate = data[data["Education"]=='Not Graduate']
graduate['LoanAmount'].plot(kind='density', label='Graduate')
not_graduate['LoanAmount'].plot(kind='density', label='Not Graduate')
plt.legend()

fig, (ax_1, ax_2, ax_3) = plt.subplots(nrows = 3 , ncols = 1, figsize=(20,10))
ax_1.scatter(data['ApplicantIncome'], data['LoanAmount'])
ax_1.set_title('Applicant Income')
ax_2.scatter(data['CoapplicantIncome'], data['LoanAmount'])
ax_2.set_title('Coapplicant Income')
data['TotalIncome'] = np.nan
data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']
ax_3.scatter(data['TotalIncome'], data['LoanAmount'])
ax_3.set_title('Total Income')