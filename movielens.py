
# coding: utf-8

# In[1]:


import pandas as pd

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table(u'F:/电影/练数成金/pydata-book-master/ch02/movielens/users.dat', sep='::', header=None, names=unames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table(u'F:/电影/练数成金/pydata-book-master/ch02/movielens/ratings.dat', sep='::', header=None, names=rnames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table(u'F:/电影/练数成金/pydata-book-master/ch02/movielens/movies.dat', sep='::', header=None, names=mnames)


# In[2]:


users[:5]


# In[3]:


ratings[:5]


# In[4]:


movies[:5]


# In[5]:


data = pd.merge(pd.merge(users,ratings),movies)
data[:5]


# In[7]:


data.iloc[0]


# In[9]:


mean_rating = data.pivot_table('rating',index = 'title',columns='gender',aggfunc='mean')
mean_rating[:5]


# In[10]:


rating_by_title = data.groupby('title').size()
rating_by_title[:10]


# In[12]:


top_female_ratings = mean_rating.sort_values(by='F',ascending=False)
top_female_ratings[:10]


# In[13]:


mean_rating['diff'] = mean_rating['M'] - mean_rating['F']
sorted_by_diff = mean_rating.sort_values(by='diff')
sorted_by_diff[:10]


# In[14]:


sorted_by_diff[::-1][:15]


# In[20]:


ratings_by_title = data.groupby('title').size()
rating_std_by_title = data.groupby('title')['rating'].std()
active_titles = ratings_by_title.index[ratings_by_title >= 250]
rating_std_by_title = rating_std_by_title.ix[active_titles]


# In[21]:


rating_std_by_title[:10]

