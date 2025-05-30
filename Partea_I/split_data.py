import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split

data_df = pd.read_csv('data.csv')

# # eliminam 'real_estate' nu avem nevoie de el
# data_df = data_df.drop(columns=['real_estate'])

# unificam late_30_59, late_60_89, late_90 in late pentru a crea o valoare categoriala

def late(row):
    if row['late_90'] > 0:
        return "severe"
    elif row['late_60_89'] > 0:
        return "medium"
    elif row['late_30_59'] > 0:
        return "low"
    else:
        return "none"

data_df["late"] = data_df.apply(late, axis=1)

data_df = data_df.drop(columns=['late_30_59', 'late_60_89', 'late_90'])

# # adaugam zgomot la monthly income
# np.random.seed(42)
# noise = np.random.normal(loc=0, scale=100, size=data_df['monthly_inc'].shape)
# data_df['monthly_inc'] = data_df['monthly_inc'] + noise

# calculam income per dependent
data_df['inc_per_dep'] = data_df['monthly_inc'] / (data_df['dependents'] + 1)

# simulez date lipsa doar unde dependents = 0
np.random.seed(0)
sim_lipsa = np.random.rand(len(data_df)) < 0.1 # 10% din randuri
for i in range(len(data_df)):
    if data_df['dependents'][i] == 0 and sim_lipsa[i]:
        data_df.loc[i, 'dependents'] = np.nan

# adaugam o noua coloana calculata din dependents
def dependents(row):
    if row['dependents'] == 0:
        return "single"
    elif row['dependents'] == 1:
        return "couple"
    elif row['dependents'] == 2:
        return "family"
    elif row['dependents'] == 3:
        return "2_kids"
    elif pd.isna(row['dependents']):
        return "unknown"
    else:
        return "extended_family"
    
# add a column with modified dependents as "relationship_status"
data_df["relationship_status"] = data_df.apply(dependents, axis=1)
# remove the original dependents column
data_df = data_df.drop(columns=['dependents'])

# il punem pe dlq pe ultima coloana
dlq_2yrs = data_df.pop('dlq_2yrs')
data_df['dlq_2yrs'] = dlq_2yrs

# impartim in 2 fisiere, unu ce contine dlq_2yrs = 0 si altul dlq_2yrs = 1
# data_0 = data_df[data_df['dlq_2yrs'] == 0]
# data_1 = data_df[data_df['dlq_2yrs'] == 1]

# train_0, test_0 = train_test_split(data_0, test_size=0.1, random_state=42, stratify=data_0['dlq_2yrs'])
# train_1, test_1 = train_test_split(data_1, test_size=0.1, random_state=42, stratify=data_1['dlq_2yrs'])

# train_0.to_csv('train_0.csv', index=False)
# train_1.to_csv('train_1.csv', index=False)
# test_0.to_csv('test_0.csv', index=False)
# test_1.to_csv('test_1.csv', index=False)

train, test = train_test_split(data_df, test_size=0.1, random_state=42, stratify=data_df['dlq_2yrs'])
train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

# show data types
# print(train_0.dtypes)