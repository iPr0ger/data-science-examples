# Plotting Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


data = pd.read_csv('heart.csv')
# print(data.head())
# print(data.info())
# print(data.describe())

# Correlations
sns.heatmap(data.corr(), annot=True, linewidth=2)

data.drop('target', axis=1).corrwith(data.target).plot(kind='bar', grid=True, figsize=(20, 10),
                                                       title="Correlation with the target feature")
plt.tight_layout()


categorical_val = []
continous_val = []
for column in data.columns:
    print("--------------------")
    print(f"{column} : {data[column].unique()}")
    if len(data[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)


categorical_val.remove('target')
dfs = pd.get_dummies(data, columns=categorical_val)


sc = StandardScaler()
col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dfs[col_to_scale] = sc.fit_transform(dfs[col_to_scale])


X = dfs.drop('target', axis=1)
y = dfs.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Models
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred1 = knn.predict(X_test)
print(accuracy_score(y_test, y_pred1))


# Hyperparameter Optimization
test_score = []
neighbors = range(1, 25)

for k in neighbors:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    test_score.append(accuracy_score(y_test, model.predict(X_test)))


plt.figure(figsize=(18, 8))
plt.plot(neighbors, test_score, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()
plt.tight_layout()

new_knn = KNeighborsClassifier(n_neighbors=19)
new_knn.fit(X_train, y_train)
y_pred2 = new_knn.predict(X_test)
print(accuracy_score(y_test, y_pred2))


# Random forest
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred3 = rfc.predict(X_test)
print(accuracy_score(y_test, y_pred3))


## Hyperparameter Optimization
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)

params2 = {
    'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': max_depth,
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

new_rfc = RandomForestClassifier(random_state=42)
rfcs = RandomizedSearchCV(estimator=new_rfc, param_distributions=params2, n_iter=100, cv=5,
                          verbose=2, random_state=42, n_jobs=-1)
rfcs.fit(X_train,y_train)
print(rfcs.best_estimator_)
y_pred4 = rfcs.predict(X_test)
print(accuracy_score(y_test, y_pred4))


# XGB
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train, y_train)
y_pred5 = xgb.predict(X_test)
print(accuracy_score(y_test,y_pred5))


# CatBoost
model4 = CatBoostClassifier(random_state=42)
model4.fit(X_train,y_train)
y_pred6 = model4.predict(X_test)
print(accuracy_score(y_test, y_pred6))


plt.show()