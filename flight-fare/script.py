import pandas as pd
import numpy as np
import string

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import seaborn as sns
import matplotlib.pyplot as plt


train_df = pd.read_excel("Data_Train.xlsx")
# print(train_df.info())
# print(train_df.describe())

test_df = pd.read_excel("Test_set.xlsx")
# print(test_df.info())
# print(test_df.describe())

train_df.dropna(inplace=True)
train_df.drop_duplicates(keep='first', inplace=True)

# Duration - convert hours in minutes
train_df['Duration'] = train_df['Duration'].str.replace("h", "*60").str.replace(' ', '+').str.replace('m', '*1').apply(eval)
test_df['Duration'] = test_df['Duration'].str.replace("h", "*60").str.replace(' ', '+').str.replace('m', '*1').apply(eval)

# Date_of_Journey
train_df['Journey_day'] = train_df['Date_of_Journey'].str.split('/').str[0].astype(int)
train_df['Journey_month'] = train_df['Date_of_Journey'].str.split('/').str[1].astype(int)
train_df.drop(['Date_of_Journey'], axis=1, inplace=True)

# Dep_time
train_df['Dep_hour'] = pd.to_datetime(train_df['Dep_Time']).dt.hour
train_df['Dep_min'] = pd.to_datetime(train_df['Dep_Time']).dt.minute
train_df.drop(['Dep_Time'], axis=1, inplace=True)

# Arrival time
train_df['Arrival_hour'] = pd.to_datetime(train_df['Arrival_Time']).dt.hour
train_df['Arrival_min'] = pd.to_datetime(train_df['Arrival_Time']).dt.minute
train_df.drop(['Arrival_Time'], axis=1, inplace=True)


# Date_of_Journey
test_df['Journey_day'] = test_df['Date_of_Journey'].str.split('/').str[0].astype(int)
test_df['Journey_month'] = test_df['Date_of_Journey'].str.split('/').str[1].astype(int)
test_df.drop(['Date_of_Journey'], axis=1, inplace=True)

# Dep_time
test_df['Dep_hour'] = pd.to_datetime(test_df['Dep_Time']).dt.hour
test_df['Dep_min'] = pd.to_datetime(test_df['Dep_Time']).dt.minute
test_df.drop(['Dep_Time'], axis=1, inplace=True)

# Arrival time
test_df['Arrival_hour'] = pd.to_datetime(test_df['Arrival_Time']).dt.hour
test_df['Arrival_min'] = pd.to_datetime(test_df['Arrival_Time']).dt.minute
test_df.drop(['Arrival_Time'], axis=1, inplace=True)


# Visualize the correlation
sns.heatmap(train_df.corr(), annot=True)


# Dropping the price
data = train_df.drop(['Price'], axis=1)

train_categorical_data = data.select_dtypes(exclude=['int64', 'float', 'int32'])
train_numerical_data = data.select_dtypes(include=['int64', 'float', 'int32'])

test_categorical_data = test_df.select_dtypes(exclude=['int64', 'float', 'int32'])
test_numerical_data = test_df.select_dtypes(include=['int64', 'float', 'int32'])


# Label encoding
le = LabelEncoder()
train_categorical_data = train_categorical_data.apply(le.fit_transform)
test_categorical_data = test_categorical_data.apply(le.fit_transform)


X = pd.concat([train_categorical_data, train_numerical_data], axis=1)
y = train_df['Price']

test_set = pd.concat([test_categorical_data, test_numerical_data], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model building
params = {'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
depth = list(range(3, 30))
params_grid = dict(max_depth=depth)
tuned_params = {'n_estimators': [100, 200], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}

# Ridge regression
ridge = GridSearchCV(Ridge(), params, cv = 5, scoring='neg_mean_absolute_error', n_jobs=-1)
ridge.fit(X_train, y_train)
ridge_predict = ridge.predict(X_test)
print("Ridge regression: ", r2_score(y_test, ridge_predict))


# Lasso
lasso_regressor = GridSearchCV(Lasso(), params, cv=15, scoring='neg_mean_absolute_error', n_jobs=-1)
lasso_regressor.fit(X_train, y_train)
lasso_regressor_predict = lasso_regressor.predict(X_test)
print("Lasso: ", r2_score(y_test, lasso_regressor_predict))


# Decision Tree
tree = GridSearchCV(DecisionTreeRegressor(), params_grid, cv=10)
tree.fit(X_train, y_train)
tree_predict = tree.predict(X_test)
print("Decision tree: ", r2_score(y_test, tree_predict))


# Random forest
random_regressor = RandomizedSearchCV(RandomForestRegressor(), tuned_params, n_iter=20,
                                      scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
random_regressor.fit(X_train, y_train)
random_regressor_predict = random_regressor.predict(X_test)
print("Random forest: ", r2_score(y_test, random_regressor_predict))

plt.show()

