
import numpy as np
new_record = [[50,  9,  4,  1,  0,  0, 40,  0]]
data = np.genfromtxt(path, delimiter=",", skip_header=1)
census = np.concatenate((data, new_record), axis=0)
print(census)

new_record = [[50,  9,  4,  1,  0,  0, 40,  0]]
data = np.genfromtxt(path, delimiter=",", skip_header=1)
census = np.concatenate((data, new_record), axis=0)
age = census[:, 0]
max_age = max(age)
min_age = min(age)
age_mean = age.mean()
age_std = age.std()

def min_len (a, b, c, d, e) :
    min = a
    if b<a :
        min = b
    elif c<a :
        min = c
    elif d<a :
        min = d
    elif e<a :
        min = e
    return min

new_record = [[50,  9,  4,  1,  0,  0, 40,  0]]
data = np.genfromtxt(path, delimiter=",", skip_header=1)
census = np.concatenate((data, new_record), axis=0)
race_0 = census[census[:,2]==0]
race_1 = census[census[:,2]==1]
race_2 = census[census[:,2]==2]
race_3 = census[census[:,2]==3]
race_4 = census[census[:,2]==4]
len_0 = len(race_0)
len_1 = len(race_1)
len_2 = len(race_2)
len_3 = len(race_3)
len_4 = len(race_4)
min = min_len(len_0, len_1, len_2, len_3, len_4)
if min==len_0 :
    minority_race = 0
elif min==len_1 :
    minority_race = 1
elif min==len_2 :
    minority_race = 2
elif min==len_3 :
    minority_race = 3
elif min==len_4 :
    minority_race = 4

new_record = [[50,  9,  4,  1,  0,  0, 40,  0]]
data = np.genfromtxt(path, delimiter=",", skip_header=1)
census = np.concatenate((data, new_record), axis=0)
age = census[:,0]
senior_citizens = census[age>60]
working_hours_sum = senior_citizens.sum(axis=0)[6]
senior_citizens_len = len(senior_citizens)
avg_working_hours = working_hours_sum / senior_citizens_len
print(avg_working_hours)

new_record = [[50,  9,  4,  1,  0,  0, 40,  0]]
data = np.genfromtxt(path, delimiter=",", skip_header=1)
census = np.concatenate((data, new_record), axis=0)
high = census[census[:,1]>10]
low = census[census[:,1]<=10]
avg_pay_high = high.mean(axis=0)[7]
avg_pay_low = low.mean(axis=0)[7]
if avg_pay_high>avg_pay_low :
    print("True")
else :
    print("False")