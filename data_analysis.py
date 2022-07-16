import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# load data
path = "data.csv"
df = pd.read_csv(path,encoding='ISO-8859-1')

# Read column names from file
cols = list(pd.read_csv(path, encoding='ISO-8859-1',nrows =1))
df = pd.read_csv(path, encoding='ISO-8859-1',usecols=[i for i in cols if i not in ['id','name','release_date']])

# removing waste stuff(square bracket and quotation marks) from artist's name
df['artists'] = df['artists'].apply(lambda x: x[1:-1].replace("'", ''))
# # correcting data types
# data['release_date'] = pd.to_datetime(data['release_date'])
# spotify['year'] = pd.to_datetime(spotify['year'].apply(lambda x: str(x)+'-01-01'))

# finding correlation
corr = df.corr()
# visualizing correlaiton with heatmap
plt.figure(figsize=(11,10))
sns.heatmap(corr, vmax=1, vmin=-1, center=0,linewidth=.5,square=True, annot = True, annot_kws = {'size':8},fmt='.1f', cmap='BrBG_r')
plt.title('Correlation between features')
plt.show()

# Adding Mean & Count values to each artist
df['mean'] = df.groupby('artists')['popularity'].transform('mean')
df['count'] = df.groupby('artists')['popularity'].transform('count')
print(df.head())

# plotting
fig, ax = plt.subplots(figsize = (8, 6))
ax = sns.distplot(df['count'],norm_hist=True,kde=False,bins = 600)
ax.set_xlabel('Count of artist appearances', fontsize=12,)
ax.set_ylabel('frequency', fontsize=12,)
# ax.set_xlim(1,20)
# ax.set_xticks(range(1,21,2))
plt.grid()
plt.show()

fig, ax = plt.subplots(figsize = (8, 6))
ax = sns.distplot(df['count'],norm_hist=True,kde=False,bins = 600)
ax.set_xlabel('Count of artist appearances', fontsize=12,)
ax.set_ylabel('frequency', fontsize=12,)
ax.set_xlim(1,20)
ax.set_xticks(range(1,21,2))
plt.grid()
plt.show()


fig, ax = plt.subplots(figsize = (8, 6))
stat = df.groupby('count')['mean'].mean().to_frame().reset_index()
ax = stat.plot(x='count', y='mean', marker='.', linestyle = '', ax=ax)
ax.set_xlabel('Count of artist appearances', fontsize=12, )
ax.set_ylabel('Mean Popularity', fontsize=12, )
plt.grid()
plt.show()