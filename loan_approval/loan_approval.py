# File path of loan_prediction.csv is stored in path

import pandas as pd
bank = pd.read_csv(path)
categorical_var = bank.select_dtypes(include='object')
print(categorical_var)
numerical_var = bank.select_dtypes(include='number')
print(numerical_var)
bank = pd.read_csv(path)
banks = bank.drop(columns='Loan_ID')
print(banks.isnull().sum())
bank_mode = banks.mode().iloc[0]
banks.fillna(bank_mode, inplace=True)
num = banks.isnull().sum()
print(num)

avg_loan_amount = pd.pivot_table(banks, index=['Gender','Married','Self_Employed'], values='LoanAmount')

loan_approved_se = banks.loc[(banks["Self_Employed"]=="Yes")  & (banks["Loan_Status"]=="Y"), ["Loan_Status"]].count()
print(loan_approved_se)
loan_approved_nse = banks.loc[(banks["Self_Employed"]=="No")  & (banks["Loan_Status"]=="Y"), ["Loan_Status"]].count()
print(loan_approved_nse)

percentage_se = (loan_approved_se * 100 / 614)
percentage_se=percentage_se[0]
print(percentage_se)

percentage_nse = (loan_approved_nse * 100 / 614)
percentage_nse=percentage_nse[0]
print (percentage_nse)

loan_term = banks['Loan_Amount_Term'].apply(lambda x : int(x)/12)
big_loan_term = len(loan_term[loan_term>=25])

loan_groupby = banks.groupby('Loan_Status')[['ApplicantIncome', 'Credit_History']]
mean_values = loan_groupby.mean()