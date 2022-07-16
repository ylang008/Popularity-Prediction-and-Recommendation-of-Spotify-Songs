import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, HTML
from plotnine import *
import pydot
from plotnine import *
from tqdm import tqdm

# For transformations and predictions
from scipy.optimize import curve_fit
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import pairwise_distances
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from six import StringIO

# For scoring
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import explained_variance_score as acc_regression

# For validation
from sklearn.model_selection import train_test_split as split

from data_preprocessing import KbestFeatures,Target_Encoding,Target_Encoding_withcutoff
# sns.set_theme(style="darkgrid")

#load data
path = "data.csv"
df = pd.read_csv(path,encoding='ISO-8859-1')

# Read column names from file
cols = list(pd.read_csv(path, encoding='ISO-8859-1',nrows =1))
df = pd.read_csv(path, encoding='ISO-8859-1',usecols=[i for i in cols if i not in ['id','name','release_date']])

# Remove duplicated
df = df[~df.duplicated()==1]

# features = ['artists','danceability', 'energy', 'loudness', 'tempo','popularity','year']
# df = df[features]
df['popularity'] = df['popularity'].div(100)

#Split the data to train and test
X_train, X_test, y_train, y_test = split(df.drop('popularity', axis=1), df['popularity'], test_size = 0.2, random_state = 12345)

# pre-processing
scaler = MinMaxScaler()
# scaler = StandardScaler()
cols = [col for col in X_train.columns if col != 'artists']
X_train[cols] = scaler.fit_transform(X_train[cols])
X_test[cols] = scaler.fit_transform(X_test[cols])

X_train_ = X_train.copy()
X_test_ = X_test.copy()

# Apply target encoding on train and test seperatly
artists_transformer = Target_Encoding() # without cutoff
X_train1 = artists_transformer.fit(X_train, y_train).transform(X_train, y_train)
X_test1 = artists_transformer.transform(X_test, y_test)

artists_transformer = Target_Encoding_withcutoff(MinCnt=3,MaxCnt=600) # withcutoff
X_train2 = artists_transformer.fit(X_train_, y_train).transform(X_train_, y_train)
X_test2 = artists_transformer.transform(X_test_, y_test)


# model 1-LinearRegression
# determine the k best features
scores_1 = np.zeros((15))
scores_2 = np.zeros((15))
for k in range(15):
    new_features = KbestFeatures(X_train1,y_train,k+1)
    model = LinearRegression()
    # Fit the model and
    model.fit(X_train1[new_features], y_train)
    scores_1[k] = model.score(X_test1[new_features], y_test) * 100

for k in range(15):
    new_features = KbestFeatures(X_train2,y_train,k+1)
    model = LinearRegression()
    # Fit the model and
    model.fit(X_train1[new_features], y_train)
    scores_2[k] = model.score(X_test2[new_features], y_test) * 100

plt.figure(1)
plt.plot(np.arange(15)+1,scores_1)
plt.xlabel('k')
plt.ylabel('test scores')
plt.xticks(np.arange(0,15,2)+1)
plt.grid()
plt.title('target encoding')

plt.figure(2)
plt.plot(np.arange(15)+1,scores_2)
plt.xlabel('k')
plt.ylabel('test scores')
plt.xticks(np.arange(0,15,2)+1)
plt.grid()
plt.title('target encoding with cutoff')
plt.show()


# model 2-KNeighborsRegressor
# determine best number of neighbors
# scores1 = np.zeros((20))
# for k in range(5, 101, 5):
#     model = KNeighborsRegressor(n_neighbors=k)
#     model.fit(X_train1, y_train)
#     scores1[int(k/5)-1] = model.score(X_test1, y_test) * 100
#     print(scores1[int(k/5)-1])
#
# scores2 = np.zeros((20))
# for k in range(5, 101, 5):
#     model = KNeighborsRegressor(n_neighbors=k)
#     model.fit(X_train2, y_train)
#     scores2[int(k/5)-1] = model.score(X_test2, y_test) * 100
#     print(scores2[int(k/5)-1])
#
# plt.figure(3)
# plt.plot(np.arange(5,101,5),scores1,label='target encoding')
# plt.plot(np.arange(5,101,5),scores2,label='target encoding with cutoff')
# plt.xlabel('k')
# plt.ylabel('test score')
# # plt.xticks(np.arange(20)+1)
# plt.legend()
# plt.grid()
# plt.title('test score with k neighbors')
# plt.show()

# scores1 = np.zeros((15))
# for k in range(15):
#     new_features = KbestFeatures(X_train1,y_train,k+1)
#     model = KNeighborsRegressor(n_neighbors=35)
#     # Fit the model and
#     model.fit(X_train1[new_features], y_train)
#     scores1[k] = model.score(X_test1[new_features], y_test) * 100
#     print(scores1[k])
#
# scores2 = np.zeros((15))
# for k in range(15):
#     new_features = KbestFeatures(X_train2,y_train,k+1)
#     model = KNeighborsRegressor(n_neighbors=20)
#     # Fit the model and
#     model.fit(X_train2[new_features], y_train)
#     scores2[k] = model.score(X_test2[new_features], y_test) * 100
#     print(scores2[k])
#
# plt.figure(4)
# plt.plot(np.arange(15)+1,scores1,label='target encoding')
# plt.plot(np.arange(15)+1,scores2,label='target encoding with cutoff')
# plt.xlabel('k')
# plt.ylabel('test score')
# plt.xticks(np.arange(0,15,2)+1)
# plt.grid()
# plt.legend()
# plt.title('test score with k best features')
# plt.show()


# model 3
# determine best max_leaf_nodes
# scores1 = np.zeros((200))
# for n in range(2,200):
#     model = DecisionTreeRegressor(random_state=15, max_leaf_nodes=n)
#     model.fit(X_train1, y_train)
#     scores1[n] = model.score(X_test1, y_test) * 100
#     print(scores1[n])
#
# scores2 = np.zeros((200))
# for n in range(2,200):
#     model = DecisionTreeRegressor(random_state=15, max_leaf_nodes=n)
#     model.fit(X_train2, y_train)
#     scores2[n] = model.score(X_test2, y_test) * 100
#     print(scores2[n])
#
# plt.figure(5)
# plt.plot(np.arange(200)+1,scores1,label='target encoding')
# plt.plot(np.arange(200)+1,scores2,label='target encoding with cutoff')
# plt.xlabel('max leaf nodes')
# plt.ylabel('test score')
# # plt.xticks(np.arange(0,15,2)+1)
# plt.grid()
# plt.legend()
# plt.title('test score with different max leaf nodes')
# plt.show()

# scores1 = np.zeros((15))
# for k in range(15):
#     new_features = KbestFeatures(X_train1,y_train,k+1)
#     model = DecisionTreeRegressor(random_state=15, max_leaf_nodes=100)
#     # Fit the model and
#     model.fit(X_train1[new_features], y_train)
#     scores1[k] = model.score(X_test1[new_features], y_test) * 100
#     print(scores1[k])
#
# scores2 = np.zeros((15))
# for k in range(15):
#     new_features = KbestFeatures(X_train2,y_train,k+1)
#     model = DecisionTreeRegressor(random_state=15, max_leaf_nodes=100)
#     # Fit the model and
#     model.fit(X_train2[new_features], y_train)
#     scores2[k] = model.score(X_test2[new_features], y_test) * 100
#     print(scores2[k])
#
# plt.figure(6)
# plt.plot(np.arange(15)+1,scores1,label='target encoding')
# plt.plot(np.arange(15)+1,scores2,label='target encoding with cutoff')
# plt.xlabel('k')
# plt.ylabel('test score')
# plt.xticks(np.arange(0,15,2)+1)
# plt.grid()
# plt.legend()
# plt.title('test score with k best features')
# plt.show()

# model 4-RandomForestRegressor
# determine the k best features
# scores1 = np.zeros((15))
# for k in range(15):
#     new_features = KbestFeatures(X_train1,y_train,k+1)
#     model = RandomForestRegressor()
#     # Fit the model and
#     model.fit(X_train1[new_features], y_train)
#     scores1[k] = model.score(X_test1[new_features], y_test) * 100
#     print(scores1[k])
#
# scores2 = np.zeros((15))
# for k in range(15):
#     new_features = KbestFeatures(X_train2,y_train,k+1)
#     model = RandomForestRegressor()
#     # Fit the model and
#     model.fit(X_train2[new_features], y_train)
#     scores2[k] = model.score(X_test2[new_features], y_test) * 100
#     print(scores2[k])
#
# plt.figure(7)
# plt.plot(np.arange(15)+1,scores1,label='target encoding')
# plt.plot(np.arange(15)+1,scores2,label='target encoding with cutoff')
# plt.xlabel('k')
# plt.ylabel('test score')
# plt.xticks(np.arange(0,15,2)+1)
# plt.grid()
# plt.legend()
# plt.title('test score with k best features')
# plt.show()
#
# best_k = np.argmax(scores2)+1
# print('best K:',best_k)

# model 5-MLPRegressor
# determine the k best features
# scores1 = np.zeros((15))
# for k in range(15):
#     new_features = KbestFeatures(X_train1,y_train,k+1)
#     model = MLPRegressor(hidden_layer_sizes=(15,12,10,5,3), random_state=1, max_iter=100000,activation='logistic')
#     # Fit the model and
#     model.fit(X_train1[new_features], y_train)
#     scores1[k] = model.score(X_test1[new_features], y_test) * 100
#     print(scores1[k])
#
# scores2 = np.zeros((15))
# for k in range(15):
#     new_features = KbestFeatures(X_train2,y_train,k+1)
#     model = MLPRegressor(hidden_layer_sizes=(15,12,10,5,3), random_state=1, max_iter=100000,activation='logistic')
#     # Fit the model and
#     model.fit(X_train2[new_features], y_train)
#     scores2[k] = model.score(X_test2[new_features], y_test) * 100
#     print(scores2[k])
#
# plt.figure(8)
# plt.plot(np.arange(15)+1,scores1,label='target encoding')
# plt.plot(np.arange(15)+1,scores2,label='target encoding with cutoff')
# plt.xlabel('k')
# plt.ylabel('test score')
# plt.xticks(np.arange(0,15,2)+1)
# plt.grid()
# plt.legend()
# plt.title('test score with k best features')
# plt.show()

