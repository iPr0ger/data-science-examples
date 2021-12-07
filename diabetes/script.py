import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score


diabetes_df = pd.read_csv('diabetes.csv')
# print(diabetes_df.info())
# print(diabetes_df.describe())

# print(diabetes_df.isnull().sum())

diabetes_df_copy = diabetes_df.copy(deep = True)
diabetes_df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes_df_copy[['Glucose',
                                                                                                      'BloodPressure',
                                                                                                      'SkinThickness',
                                                                                                      'Insulin',
                                                                                                      'BMI']].replace(0,np.NaN)

# Plotting
# diabetes_df.hist(figsize=(20, 20))

# Aiming to impute NAN values for the columns in accordance with their distribution
diabetes_df_copy['Glucose'].fillna(diabetes_df_copy['Glucose'].mean(), inplace=True)
diabetes_df_copy['BloodPressure'].fillna(diabetes_df_copy['BloodPressure'].mean(), inplace=True)
diabetes_df_copy['SkinThickness'].fillna(diabetes_df_copy['SkinThickness'].median(), inplace=True)
diabetes_df_copy['Insulin'].fillna(diabetes_df_copy['Insulin'].median(), inplace=True)
diabetes_df_copy['BMI'].fillna(diabetes_df_copy['BMI'].median(), inplace=True)

# Plotting the distributions after removing the NAN values
# diabetes_df_copy.hist(figsize=(20, 20))

# scatter_matrix(diabetes_df, figsize=(20, 20))
# sns.pairplot(diabetes_df_copy, hue='Outcome')

# Correlations - initial dataset
# sns.heatmap(diabetes_df.corr(), annot=True, cmap='RdYlGn')

# Correlations after cleaning
# sns.heatmap(diabetes_df_copy.corr(), annot=True, cmap='RdYlGn')

sc = StandardScaler()
X = pd.DataFrame(sc.fit_transform(diabetes_df_copy.drop(["Outcome"], axis=1),),
                 columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                          'DiabetesPedigreeFunction', 'Age'])

y = diabetes_df_copy['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# Building a model
test_scores = []
train_scores = []

for i in range(1, 15):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)

    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))


max_train_score = max(train_scores)
train_scores_ind = [i for i, v in enumerate(train_scores) if v == max_train_score]
print('Max train score {} % and k = {}'.format(max_train_score*100, list(map(lambda x: x+1, train_scores_ind))))

# score that comes from testing on the datapoints that were split in the beginning to be used
# for testing solely
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100, list(map(lambda x: x+1, test_scores_ind))))

# In case of Classifier like KNN the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn,param_grid, cv=5)
knn_cv.fit(X, y)

print("Best Score:" + str(knn_cv.best_score_))
print("Best Parameters: " + str(knn_cv.best_params_))

plt.show()
