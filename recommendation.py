import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 50)

import numpy as np
from sklearn.preprocessing import MinMaxScaler # for normalizing data
from sklearn.cluster import KMeans

# load data
data = pd.read_csv('data.csv', encoding='ISO-8859-1')
data.rename(columns={'name':'song'},inplace=True)

# see if there are missing values
print(data.isna().sum())

# removing waste stuff(square bracket and quotation marks) from artist's name
data['artists'] = data['artists'].apply(lambda x: x[1:-1].replace("'", ''))


song_features = pd.DataFrame()

# normalizer instance
scaler = MinMaxScaler()
for col in data.iloc[:, :-1].columns:      # excluding year col i.e, of int64 type
    if data[col].dtypes in ['float64', 'int64']:
        # adding normalized col
        scaler.fit(data[[col]])
        song_features[col] = scaler.transform(data[col].values.reshape(-1, 1)).ravel()

# Find best number of clusters
# KMeans instance
# km = KMeans()
# k_rng = range(1, 1200,10)  # k value
# sse = []  # sse value for each k
# for i in k_rng:
#     km = KMeans(n_clusters=i)
#     km.fit(song_features.sample(2000))
#     # calculating sse
#     sse.append(km.inertia_)

# plt.plot(k_rng,sse)
# plt.xlabel('K value')
# plt.ylabel('SSE Error')
# plt.title('Best K value')
# # plt.ylim(0,400)
# # plt.xlim(0,100)
# plt.show()

# reading songs data
songs_df = data.copy()
# artists_df=artists_df.drop(columns=['artists','id','name','release_date','year'])
# playcount

# artists_df = artists_df.rename(columns={"count": "playCount"})

songs_df.iloc[:, [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17]] = scaler.fit_transform(songs_df.iloc[:, [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17]])
# Note that it takes times when use 1000 clusters, you can also decrease the number of clusters to save time
km = KMeans(n_clusters=1000)
songs_df['genres'] = km.fit_predict(songs_df.iloc[:, [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 15, 16, 17]])
songs_df = songs_df.iloc[:, [1, 12,13, -1]]
# print(songs_df.head())

# ramdomly create our own user list with his/her rating and add to artists data
songs_df['playCount'] = np.random.randint(0, 100, len(songs_df))
songs_df['user_id'] = np.random.randint(0, 1000, len(songs_df))
# artists_df['rating'] = np.random.randint(1,6,len(artists_df))
# print(songs_df.head())

# recommendation system
def recommend_me(user):

    # choose user top liked genres
    fav_genre = songs_df[songs_df['user_id'] == user].sort_values(by=['playCount'], ascending=False)[
                    'genres'][:5]
    fav_genre = list(dict.fromkeys(fav_genre))  # removing duplicate if exits

    # clear out the songs from list which have been listened by the user
    # listened_songs = songs_df.index[
        # songs_df['song'].isin(['Keep A Song In Your Soul', 'True House Music - Original Massive Mix'])].tolist()

    # remaining_songs = songs_df.drop(listened_songs, axis=0)
    remaining_songs = songs_df
    CanBeRecommended = remaining_songs[remaining_songs['genres'].isin(fav_genre)]

    # sort our songs whose are popular in this user favorite genre
    CanBeRecommended = CanBeRecommended.sort_values(by=['popularity'], ascending=False)[ #recommend based on popularity
                          ['artists','song', 'playCount','popularity','genres',]][:5]

    # output will contain artists, song,  playcount, popularity,genres
    return CanBeRecommended

# recommend this user some artists
print(recommend_me(12))

# check which genre is user fav and did he get same recommended
print(songs_df[songs_df.user_id == 12].sort_values(by='playCount',ascending=False)[['artists','song', 'playCount','popularity','genres',]][:5])
