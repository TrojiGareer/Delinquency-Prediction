import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split

# concatenam toate datele intr-un singur dataframe
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

data_df = pd.concat([df_train, df_test], ignore_index=True)

# b) Statistici descriptive

# variabilele numerice
# print("\nStatistici descriptive pentru variabilele numerice:\n")
# print(data_df.describe(include="all"))

data_df.describe(include="all").to_csv('description.csv')


# variabilele categorice
# print("\nStatistici descriptive pentru variabilele categorice (string):\n")
# print(data_df[['late', 'relationship_status']].describe())

print("\nStatistici descriptive pentru variabilele categorice (numerice):\n")
# transformam stringurile categorice in valori numerice
# Label Encoding
# print(data_df.head(20))
# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()
# late_encoded = encoder.fit_transform(data_df['late'])

# tratam "unknown" ca "single"

# data_df['relationship_status'] = data_df['relationship_status'].replace('unknown', 'single')
# rel_encoded = encoder.fit_transform(data_df['relationship_status'])

late_map = {
    'none': 0,
    'low': 1,
    'medium': 2,
    'severe': 3
}

data_df['late'] = data_df['late'].map(late_map)

data_df['relationship_status'] = data_df['relationship_status'].replace('unknown', 'single')

relationship_map = {
    'single': 1,
    'couple': 2,
    'family': 3,
    '2_kids': 4,
    'extended_family': 5
}

data_df['relationship_status'] = data_df['relationship_status'].map(relationship_map)

# print(data_df[['late', 'relationship_status']].describe())

print(data_df.describe())

# IQR pentru a gasi outliers
def calculate_iqr(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lwr = q1 - 1.5 * iqr
    uppr = q3 + 1.5 * iqr
    return lwr, uppr

# nu aplicam IQR pe tinta
cols = data_df.select_dtypes(include=np.number).columns.tolist()
cols.remove('dlq_2yrs')
for i in cols:
    lwr, uppr = calculate_iqr(data_df, i)
    # outliers = data_df[(data_df[i] < lwr) | (data_df[i] > uppr)]
    #  remove outliers
    data_df[i] = np.clip(data_df[i], lwr, uppr)

data_df['rev_util'] = np.where(data_df['rev_util'] > 1, data_df['rev_util'] / 100, data_df['rev_util'])

print("\nDupa aplicarea IQR\n")
print(data_df.describe())






# print(data_df.dtypes)

# Normalizare
for i in cols:
    min_val = data_df[i].min()
    max_val = data_df[i].max()
    data_df[i] = (data_df[i] - min_val) / (max_val - min_val)


print("\nDupa normalizare\n")
print(data_df.describe())

# # Standardizare

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# data_df[cols] = scaler.fit_transform(data_df[cols])
# print("\nDupa standardizare\n")
# print(data_df.describe())


# histograma pentru rev_util
bins = np.linspace(data_df['rev_util'].min(), data_df['rev_util'].max(), 50)
plt.hist(data_df['rev_util'], bins=bins, edgecolor='black')
plt.xlabel('Revolving Utilization of Balance')
plt.ylabel('Frequency')
plt.title('Histogram of Revolving Utilization of Balance')
plt.show()

# coeficient de corelatie

# heatmap
corr_coef = data_df.corr(method='pearson')


sns.heatmap(corr_coef, annot=True, cmap='coolwarm', center=0)
plt.show()

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

