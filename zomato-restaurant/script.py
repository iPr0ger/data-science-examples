import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import os
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
from geopy.geocoders import Nominatim
import dexplot as dxp
import re
import string
import nltk
import ssl
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Disable ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Download NLTK package
# nltk.download('stopwords')

df = pd.read_csv('zomato.csv')
# print(df.head())
# print(df.info())
# print(df.isnull().sum())
# print(df.describe())
# print(df.shape)

# Is booking or not and how many
# sns.countplot(df['book_table'])
# plt.title('Restaurant provides booking or not')

# Online order or not and  how many
# sns.countplot(df['online_order'])
# plt.title('Restaurant delivering or not')

# Unique values check
# print(df['listed_in(type)'].unique())

# Visualization of the following values
# dxp.count(val='listed_in(type)', data=df, figsize=(20,  12),
#           split='listed_in(city)', normalize=True)

# 93 different types of Restaurants
# print(df['rest_type'].nunique())
# print(df['rest_type'].unique())

# Unique rate data
# print(df['rate'].unique())

# Replace 'NEW' with NaN
df['rate'] = df['rate'].replace("NEW", np.nan)
df.dropna(how='any', inplace=True)

data = df
# Convert float to string
data['rate'] = data['rate'].astype(str)
data['rate'] = data['rate'].apply(lambda x: x.replace('/5', ''))
# Returning back to float
data['rate'] = data['rate'].apply(lambda x: float(x))

# Locations processing
locations = pd.DataFrame({"Name": df['location'].unique()})
# print(locations.head())
locations['Name'] = locations['Name'].apply(lambda x: "Bangaluru " + str(x))
lat_lon = []
geolocator = Nominatim(user_agent='app')
for location in locations['Name']:
    location = geolocator.geocode(location)
    if location is None:
        lat_lon.append(np.nan)
    else:
        geo = (location.latitude, location.longitude)
        lat_lon.append(geo)

locations['geo_loc'] = lat_lon
locations.to_csv('locations.csv', index=False)
locations['Name'] = locations['Name'].apply(lambda x: x.replace("Bangaluru", "")[1:])


# Defining a base_map function
def generate_base_map():
    default_location = [12.97, 77.59]
    default_zoom_start = 12
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map


Rest_locations = pd.DataFrame(df['location'].value_counts().reset_index())
Rest_locations.columns = ['Name', 'count']
Rest_locations = Rest_locations.merge(locations, on='Name', how='left').dropna()

lat, lon = zip(*np.array(Rest_locations['geo_loc']))
Rest_locations['lat'] = lat
Rest_locations['lon'] = lon
basemap = generate_base_map()
HeatMap(Rest_locations[['lat', 'lon', 'count']].values.tolist(), radius=15).add_to(basemap)


def produce_data(col, name):
    p_data = pd.DataFrame(df[df[col] == name].groupby(['location'], as_index=False)['url'].agg('count'))
    p_data.columns = ['Name', 'count']
    # print(p_data.head())
    p_data = p_data.merge(locations, on='Name', how='left').dropna()
    p_data['lan'], p_data['lon'] = zip(*data['geo_loc'].values)
    return p_data.drop(['geo_loc'], axis=1)


# food = produce_data('cuisines', 'South India')
# food_base_map = generate_base_map()
# HeatMap(food[['lat', 'lon', 'count']].values.tolist(), radius=15).add_to(food_base_map)

df['online_order'].replace(('Yes', 'No'), (True, False), inplace=True)
df['book_table'].replace(('Yes', 'No'), (True, False), inplace=True)


def encode(df):
    for column in df.columns[~df.columns.isin(['rate', 'approx_cost(for two people)', 'votes'])]:
        df[column] = df[column].factorize()[0]
    return df


en_df = encode(df.copy())

en_df['approx_cost(for two people)'] = en_df['approx_cost(for two people)'].astype(str)
en_df['approx_cost(for two people)'] = en_df['approx_cost(for two people)'].apply(lambda x: x.replace(',', '.'))
en_df['approx_cost(for two people)'] = en_df['approx_cost(for two people)'].astype(float)

# Get the correlation between params
corr = en_df.corr()
# sns.heatmap(corr, annot=True)

X = en_df.drop(['rate'], axis=1)
y = en_df['rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# reg_model = LinearRegression()
# reg_model.fit(X_train, y_train)
# y_pred = reg_model.predict(X_test)

# Check the score
# print(r2_score(y_test, y_pred))


CBR_model = CatBoostRegressor(
    n_estimators=200,
    loss_function='MAE',
    eval_metric='RMSE'
)
CBR_model.fit(X_train, y_train)
CBR_y_pred = CBR_model.predict(X_test)
# print(r2_score(y_test, CBR_y_pred))


RF_model = RandomForestRegressor(
    n_estimators=600, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=0.0001,
    min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
    verbose=0, warm_start=FileExistsError, ccp_alpha=0.0, max_samples=None
)

RF_model.fit(X_train, y_train)
RF_y_predict = RF_model.predict(X_test)
# print(r2_score(y_test, RF_y_predict))


DT_model = DecisionTreeRegressor(
    criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=0.00011,
    min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_impurity_split=None, presort='deprecated', ccp_alpha=0.0
)
DT_model.fit(X_train, y_train)
DT_y_predict = DT_model.predict(X_test)
print(r2_score(y_test, DT_y_predict))




plt.show()

