import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import GridSearchCV, train_test_split, ShuffleSplit, cross_val_score
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor

matplotlib.rcParams["figure.figsize"] = (20, 10)

df = pd.read_csv('data.csv')
# print(df.head())
# print(df.info())
# print(df.shape)
# print(df.describe())
# print(df.isnull().sum())

# Performing Group by operation on Area Type
df.groupby("area_type")["area_type"].agg("count")

# Remove less important features
df = df.drop(["availability", "area_type", "society", "balcony"], axis="columns")
# print(df.head())

# Drop N/A values
df = df.dropna()
# print(df.isnull().sum())

# Feature engineering part
# print(df["size"].unique())
df["BHK"] = df["size"].apply(lambda x: int(x.split(" ")[0]))

# print(df["total_sqft"].unique())


# Explore total_sqft feature
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


df[~df["total_sqft"].apply(is_float)].head(10)
# print(var)


def convert_sqft_to_number(x):
    tokens = x.split("-")
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None


df = df.copy()
df["total_sqft"] = df["total_sqft"].apply(convert_sqft_to_number)
# print(df.head(10))

df = df.copy()
df["price_per_sqft"] = df["price"] * 100000 / df["total_sqft"]
# print(df.head())

df["location"] = df["location"].apply(lambda x: x.strip())
location_stats = df["location"].value_counts(ascending=False)
# print(location_stats)
location_less_than_10 = location_stats[location_stats <= 10]
df["location"] = df["location"].apply(lambda x: "other" if x in location_less_than_10 else x)

# Check the Sqft values
df = df[~(df["total_sqft"]/df["BHK"] < 300)]


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby("location"):
        m = np.mean(subdf["price_per_sqft"])
        st = np.std(subdf["price_per_sqft"])
        reduced_df = subdf[(subdf["price_per_sqft"] > (m - st)) & (subdf["price_per_sqft"] <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


df = remove_pps_outliers(df)
# print(df.shape)


# Visualization
def plot_scatter_chart(df, location):
    bhk2 = df[(df["location"] == location) & (df["BHK"] == 2)]
    bhk3 = df[(df["location"] == location) & (df["BHK"] == 3)]
    matplotlib.rcParams['figure.figsize'] = (8, 6)
    plt.scatter(bhk2["total_sqft"], bhk2["price"], color="blue", label="2 BHK", s=50)
    plt.scatter(bhk3["total_sqft"], bhk3["price"], color="green", label="3 BHK", s=50, marker="+")
    plt.xlabel("Total Sqft")
    plt.ylabel("Price")
    plt.title("Location")
    plt.legend()


# plot_scatter_chart(df, "Rajaji Nagar")
"""
plt.hist(df["price_per_sqft"], rwidth=0.8)
plt.xlabel("Price per Sqft")
plt.ylabel("Count")
plt.title("Prices per Sqft")
"""

# Using one hot encoding for the location param
dummies = pd.get_dummies(df["location"])
# print(dummies.head())

# Concat two dataframes
df = pd.concat([df, dummies.drop("other", axis=1)], axis=1)
df = df.drop(["location"], axis=1)
# print(df.head())

X = df.drop(["price", "size", "price_per_sqft"], axis=1)
y = df["price"]

# print(X.shape)
# print(y.shape)

# Building a model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr_clt = LinearRegression()
lr_clt.fit(X_train, y_train)

# accuracy - 0.7900425477740703
# print(lr_clt.score(X_test, y_test))

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# print(cross_val_score(LinearRegression(), X, y, cv=cv))


# Usage of GridSearchCv to find a better model
def find_best_model(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['mse', 'friedman_mse'],
                'splitter': ['best', 'random']
            }
        }
    }

    scores = []

    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])


best_models = find_best_model(X, y)
# print(best_models)


def predict_price(location, sqft, bath, bk):
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bk

    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clt.predict([x])[0]


pred1 = predict_price('1st Phase JP Nagar', 1000, 2, 2)
# print(pred1)
pred2 = predict_price('1st Phase JP Nagar', 2000, 3, 3)
# print(pred2)


plt.show()
