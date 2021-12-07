import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from category_encoders import OrdinalEncoder

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('adult.csv')
# print(df.head())
print(df.info())
# print(df.describe())

# remove unnecessary columns
df.drop([' 2174', ' 0', ' 40'], axis='columns', inplace=True)

df.columns = ['Age', 'Type_of_Owner', 'id', 'Education', 'No_of_Projects_Done', 'Marital_Status', 'Job_Designation',
              'Family_Relation', 'Race', 'Gender', 'Country', 'Salary']
# print(df.head())

X = df.drop(['Age', 'id', 'Salary', 'Country', 'Family_Relation'], axis=1)
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

encoder = OrdinalEncoder(cols=['Education', 'Type_of_Owner', 'Gender', 'Job_Designation',
                               'Marital_Status', 'No_of_Projects_Done', 'Race'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf.fit(X_train, y_train)
clf_pred = clf.predict(X_test)
# print(round(clf.score(X_train, y_train), 2))
# print(round(clf.score(X_test, y_test), 2))

# Standardization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
# print(round(log_reg.score(X_train, y_train), 2))
# print(round(log_reg.score(X_test, y_test), 2))


random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
random_forest_pred = random_forest.predict(X_test)
# print(round(random_forest.score(X_train, y_train), 2))
# print(round(random_forest.score(X_test, y_test), 2))


k_n = KNeighborsClassifier()
k_n.fit(X_train, y_train)
k_n_pred = k_n.predict(X_test)
print(round(k_n.score(X_train, y_train), 2))
print(round(k_n.score(X_test, y_test), 2))



plt.show()