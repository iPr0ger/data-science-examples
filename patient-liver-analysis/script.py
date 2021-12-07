import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import warnings

warnings.filterwarnings('ignore')


df = pd.read_csv('indian_liver_patient.csv')
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# sns.catplot(x="Age", y="Gender", hue="Dataset", data=df)
# sns.jointplot("Total_Bilirubin", "Direct_Bilirubin", data=df, kind='reg')
# sns.jointplot("Aspartate_Aminotransferase", "Alamine_Aminotransferase", data=df, kind='reg')


# Feature engineering
df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].mean())
df = pd.concat([df, pd.get_dummies(df['Gender'], prefix='Gender')], axis=1)

X = df.drop(['Gender', 'Dataset'], axis=1)
y = df['Dataset']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
print(round(log_reg.score(X_train, y_train), 2))
print(round(log_reg.score(X_test, y_test), 2))

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
gaussian_pred = gaussian.predict(X_test)
print(round(gaussian.score(X_train, y_train), 2))
print(round(gaussian.score(X_test, y_test), 2))

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
random_forest_pred = random_forest.predict(X_test)
print(round(random_forest.score(X_train, y_train), 2))
print(round(random_forest.score(X_test, y_test), 2))

plt.show()