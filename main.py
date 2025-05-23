import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

data_df = pd.read_csv('data.csv')

# coeficient de corelatie

corr_coef = data_df.corr(method='pearson')

# print(data_df['monthly_inc'].describe())


# heatmap

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

list_late = [data_df['late_30_59'], data_df['late_60_89'], data_df['late_90']]

for i in range(len(data_df['late_30_59'])):
    if data_df['late_30_59'][i] > 1:
        data_df.loc[i, 'late_30_59'] = 1

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

from sklearn.model_selection import train_test_split


x_train, y_train, x_test, y_test = train_test_split(data_df.drop(columns=['dlq_2yrs']), data_df['dlq_2yrs'],
                                                    test_size=0.05, random_state=42)

