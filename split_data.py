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
data_0 = data_df[data_df['dlq_2yrs'] == 0]
data_1 = data_df[data_df['dlq_2yrs'] == 1]

train_0, test_0 = train_test_split(data_0, test_size=0.1, random_state=42, stratify=data_0['dlq_2yrs'])
train_1, test_1 = train_test_split(data_1, test_size=0.1, random_state=42, stratify=data_1['dlq_2yrs'])

train_0.to_csv('train_0.csv', index=False)
train_1.to_csv('train_1.csv', index=False)
test_0.to_csv('test_0.csv', index=False)
test_1.to_csv('test_1.csv', index=False)

# show data types
# print(train_0.dtypes)

# train_df, test_df = train_test_split(data_df, test_size=0.1, random_state=42, stratify=data_df['dlq_2yrs'])

# train_df.to_csv('train.csv', index=False)
# test_df.to_csv('test.csv', index=False)

# list_late = [data_df['late_30_59'], data_df['late_60_89'], data_df['late_90']]

# for i in range(len(data_df['late_30_59'])):
#     if data_df['late_30_59'][i] > 1:
#         data_df.loc[i, 'late_30_59'] = 1

# impartim data.csv in 2 fisiere, unu ce contine dlq_2yrs = 0 si altul dlq_2yrs = 1

# data_df_0 = data_df[data_df['dlq_2yrs'] == 0]
# data_df_1 = data_df[data_df['dlq_2yrs'] == 1]
# data_df_0.to_csv('data_0.csv', index=False)
# data_df_1.to_csv('data_1.csv', index=False)
# data_df_0 = pd.read_csv('data_0.csv')
# data_df_1 = pd.read_csv('data_1.csv')
# data_df_0 = data_df_0.sample(frac=0.1, random_state=42)
# data_df_1 = data_df_1.sample(frac=0.1, random_state=42)
# data_df = pd.concat([data_df_0, data_df_1], ignore_index=True)


# coeficient de corelatie

# corr_coef = data_df.corr(method='pearson')
# print(data_df.dtypes)

# print(data_df['monthly_inc'].describe())


# heatmap

# sns.heatmap(corr_coef, annot=True, cmap='coolwarm', center=0)
# plt.show()

# print(np.mean(data_df['monthly_inc']))
# print(np.median(data_df['monthly_inc']))

# since salaries are so far apart we will group them in bins
# from min to max we split them in 10 bins


# trebuie sa fac asta pentru toate variabilele
# bins = np.linspace(data_df['monthly_inc'].min(), 28000, 50)
# plt.hist(data_df['monthly_inc'], bins=bins, edgecolor='black')
# plt.xlabel('Monthly Income')
# plt.ylabel('Frequency')
# plt.title('Histogram of Monthly Income')
# plt.show()


# sns.countplot(x='late_30_59', data=data_df)
# plt.xlabel('Late Payments')
# plt.ylabel('Frequency')
# plt.title('Count of Late Payments')
# plt.show()

# sns.countplot(x='late_60_89', data=data_df)
# plt.xlabel('Late Payments')
# plt.ylabel('Frequency')
# plt.title('Count of Late Payments')
# plt.show()
# sns.countplot(x='late_90', data=data_df)
# plt.xlabel('Late Payments')
# plt.ylabel('Frequency')
# plt.title('Count of Late Payments')
# plt.show()

# IQR pentru a gasi outliers
# sau Z-score

# q1 = 


# scatter plot

# plt.scatter(data_df['monthly_inc'], data_df['dlq_2yrs'])
# plt.xlabel('Monthly Income')
# plt.ylabel('Delinquency in 2 Years')
# plt.title('Scatter Plot of Monthly Income vs. Delinquency in 2 Years')
# plt.legend()
# plt.show()
# # plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')data_df['monthly_inc'].quantile(0.25)

# train for classification

# from sklearn.model_selection import train_test_split


# x_train, y_train, x_test, y_test = train_test_split(data_df.drop(columns=['dlq_2yrs']), data_df['dlq_2yrs'],
#                                                     test_size=0.05, random_state=42)

