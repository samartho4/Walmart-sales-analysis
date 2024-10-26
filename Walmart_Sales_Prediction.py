
# Importing the basic libraries
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from statsmodels.formula import api
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 6]
import warnings 
warnings.filterwarnings('ignore')

# Importing the dataset
# Importing the dataset
df = pd.read_csv('C:\\Users\\samar\\OneDrive\\Desktop\\Walmart_Sales\\Walmart.csv')

display(df.head())

original_df = df.copy(deep=True)
print('\n\033[1mInference:\033[0m The Datset consists of {} features & {} samples.'.format(df.shape[1], df.shape[0]))

# Reframing the columns
df.Date = pd.to_datetime(df.Date, dayfirst=True)
df['weekday'] = df.Date.dt.weekday
df['month'] = df.Date.dt.month
df['year'] = df.Date.dt.year
df.drop(['Date'], axis=1, inplace=True)


target = 'Weekly_Sales'
features = [i for i in df.columns if i not in [target]]
original_df = df.copy(deep=True)
df.head()

# Checking the dtypes of all the columns
df.info()

# Checking number of unique rows in each feature
df.nunique().sort_values()

nu = df[features].nunique().sort_values()
nf = []
cf = []
nnf = 0
ncf = 0  # numerical & categorical features

for i in range(df[features].shape[1]):
    if nu.values[i] <= 45:
        cf.append(nu.index[i])
    else:
        nf.append(nu.index[i])

print('\n\033[1mInference:\033[0m The Datset has {} numerical & {} categorical features.'.format(len(nf), len(cf)))
display(df.describe())

# Plotting the distribution of the target variable
plt.figure(figsize=[8, 4])
sns.distplot(df[target], color='g', hist_kws=dict(edgecolor="black", linewidth=2), bins=30)
plt.title('Target Variable Distribution - Median Value of Homes ($1Ms)')
plt.show()

# Visualising the categorical features
print('\033[1mVisualising Categorical Features:'.center(100))
n = 2
plt.figure(figsize=[15, 3 * math.ceil(len(cf) / n)])
for i in range(len(cf)):
    if df[cf[i]].nunique() <= 8:
        plt.subplot(math.ceil(len(cf) / n), n, i + 1)
        sns.countplot(df[cf[i]])
    else:
        plt.subplot(3, 1, i - 1)
        sns.countplot(df[cf[i]])
plt.tight_layout()
plt.show()

# Visualising the numeric features
print('\033[1mNumeric Features Distribution'.center(130))
n = 4
plt.figure(figsize=[15, 6 * math.ceil(len(nf) / n)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf) / 3), n, i + 1)
    sns.distplot(df[nf[i]], hist_kws=dict(edgecolor="black", linewidth=2), bins=10, color=list(np.random.randint([255, 255, 255]) / 255))
plt.tight_layout()
plt.show()

plt.figure(figsize=[15, 6 * math.ceil(len(nf) / n)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf) / 3), n, i + 1)
    df.boxplot(nf[i])
plt.tight_layout()
plt.show()

# Understanding the relationship between all the features
g = sns.pairplot(df)
plt.title('Pairplots for all the Feature')
g.map_upper(sns.kdeplot, levels=4, color=".2")
plt.show()

# Removal of any Duplicate rows (if any)
counter = 0
rs, cs = original_df.shape
df.drop_duplicates(inplace=True)

if df.shape == (rs, cs):
    print('\n\033[1mInference:\033[0m The dataset doesn\'t have any duplicates')
else:
    print(f'\n\033[1mInference:\033[0m Number of duplicates dropped/fixed ---> {rs - df.shape[0]}')

# Check for empty elements
nvc = pd.DataFrame(df.isnull().sum().sort_values(), columns=['Total Null Values'])
nvc['Percentage'] = round(nvc['Total Null Values'] / df.shape[0], 3) * 100
print(nvc)

# Converting categorical Columns to Numeric
df3 = df.copy()
ecc = nvc[nvc['Percentage'] != 0].index.values
fcc = [i for i in cf if i not in ecc]
oh = True
dm = True
for i in fcc:
    if df3[i].nunique() == 2:
        if oh == True:
            print("\033[1mOne-Hot Encoding on features:\033[0m")
        print(i)
        oh = False
        df3[i] = pd.get_dummies(df3[i], drop_first=True, prefix=str(i))
    if df3[i].nunique() > 2:
        if dm == True:
            print("\n\033[1mDummy Encoding on features:\033[0m")
        print(i)
        dm = False
        df3 = pd.concat([df3.drop([i], axis=1), pd.DataFrame(pd.get_dummies(df3[i], drop_first=True, prefix=str(i)))], axis=1)

# Removal of outliers
df1 = df3.copy()
features1 = nf
for i in features1:
    Q1 = df1[i].quantile(0.25)
    Q3 = df1[i].quantile(0.75)
    IQR = Q3 - Q1
    df1 = df1[df1[i] <= (Q3 + (1.5 * IQR))]
    df1 = df1[df1[i] >= (Q1 - (1.5 * IQR))]
    df1 = df1.reset_index(drop=True)
display(df1.head())
print('\n\033[1mInference:\033[0m\nBefore removal of outliers, The dataset had {} samples.'.format(df3.shape[0]))
print('After removal of outliers, The dataset now has {} samples.'.format(df1.shape[0]))

# Final Dataset size after performing Preprocessing
df = df1.copy()
df.columns = [i.replace('-', '_') for i in df.columns]
plt.title('Final Dataset')
plt.pie([df.shape[0], original_df.shape[0] - df.shape[0]], radius=1, labels=['Retained', 'Dropped'], counterclock=False, autopct='%1.1f%%', pctdistance=0.9, explode=[0, 0], shadow=True)
plt.pie([df.shape[0]], labels=['100%'], labeldistance=-0, radius=0.78)
plt.show()
print(f'\n\033[1mInference:\033[0m After the cleanup process, {original_df.shape[0] - df.shape[0]} samples were dropped, while retaining {round(100 - (df.shape[0] * 100 / (original_df.shape[0])), 2)}% of the data.')

# Splitting the data into training & testing sets
m = []
for i in df.columns.values:
    m.append(i.replace(' ', '_'))
df.columns = m
X = df.drop([target], axis=1)
Y = df[target]
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=100)
Train_X.reset_index(drop=True, inplace=True)
print('Original set  ---> ', X.shape, Y.shape, '\nTraining set  ---> ', Train_X.shape, Train_Y.shape, '\nTesting set   ---> ', Test_X.shape, '', Test_Y.shape)

# Feature Scaling (Standardization)
std = StandardScaler()
print('\033[1mStandardardization on Training set'.center(120))
Train_X_std = std.fit_transform(Train_X)
Train_X_std = pd.DataFrame(Train_X_std, columns=X.columns)
display(Train_X_std.describe())

print('\n', '\033[1mStandardardization on Testing set'.center(120))
Test_X_std = std.transform(Test_X)
Test_X_std = pd.DataFrame(Test_X_std, columns=X.columns)
display(Test_X_std.describe())
