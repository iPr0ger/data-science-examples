import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from datetime import datetime

train_df = pd.read_csv('train.csv')
# print(train_df.info())
# print(train_df.describe())

test_df = pd.read_csv('test.csv')
# print(test_df.info())
# print(test_df.describe())

stores_df = pd.read_csv('stores.csv')
# print(stores_df.info())
# print(stores_df.describe())

features_df = pd.read_csv('features.csv')
# print(features_df.info())
# print(features_df.describe())

# Merging
dataset = features_df.merge(stores_df, how='inner', on='Store')

# Convert the dates
dataset['Date'] = pd.to_datetime(dataset['Date'])
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])

dataset['Week'] = dataset['Date'].dt.isocalendar().week
dataset['Year'] = dataset['Date'].dt.isocalendar().year
# print(dataset.info())

# Merging with train_df
train_merge = train_df.merge(dataset, how='inner', on=['Store', 'Date', 'IsHoliday']).sort_values(
    by=['Store', 'Dept', 'Date']).reset_index(drop=True)
test_merge = test_df.merge(dataset, how='inner', on=['Store', 'Date', 'IsHoliday']).sort_values(
    by=['Store', 'Dept', 'Date']).reset_index(drop=True)
# print(train_merge.info())
# print(test_merge.info())

train_merge = train_merge.drop(['Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis=1)
test_merge = test_merge.drop(['Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'], axis=1)

# print(train_merge.info())


# Dividing the data
X = train_merge[['Store', 'Dept', 'IsHoliday', 'Size', 'Week', 'Year']]
y = train_merge['Weekly_Sales']

# Preparation of the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Ridge Regression
params = {'alpha': [0.00001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
ridge_regression = GridSearchCV(Ridge(), params, cv=7, scoring='neg_mean_absolute_error', n_jobs=-1)
ridge_regression.fit(X_train, y_train)
ridge_regression_train_pred = ridge_regression.predict(X_train)
ridge_regression_test_pred = ridge_regression.predict(X_test)
# print(r2_score(y_test.values, ridge_regression_test_pred))


# Lasso regression
lasso_regression = GridSearchCV(Lasso(), params, cv=15, scoring='neg_mean_absolute_error', n_jobs=-1)
lasso_regression.fit(X_train, y_train)
lasso_regression_test_pred = lasso_regression.predict(X_test)
# print(r2_score(y_test.values, lasso_regression_test_pred))


# Decision Trees
depth = list(range(3, 30))
params_grid = dict(max_depth=depth)
tree = GridSearchCV(DecisionTreeRegressor(), params_grid, cv=10)
tree.fit(X_train, y_train)
tree_test_pred = tree.predict(X_test)
# print(r2_score(y_test.values, tree_test_pred))


# Random forest
tuned_params = {'n_estimators': [100, 200], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
random_regressor = RandomizedSearchCV(RandomForestRegressor(), tuned_params, n_iter=3, scoring='neg_mean_absolute_error',
                                      n_jobs=-1)
random_regressor.fit(X_train, y_train)
random_regressor_test_pred = random_regressor.predict(X_test)
print(r2_score(y_test.values, random_regressor_test_pred))

plt.show()
