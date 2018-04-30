
# coding: utf-8

# In[1]:


import numpy as np
from numpy.random import randn


# In[2]:


arr = randn(6)
arr


# In[3]:


arr.sort()


# In[4]:


arr


# In[6]:


arr = randn(3,4)
arr


# In[9]:


arr[:,0].sort()
arr


# In[10]:


arr = randn(5)
arr


# In[11]:


np.sort(arr)


# In[12]:


arr


# In[13]:


arr = randn(3,5)
arr.sort(axis = 1)


# In[14]:


arr


# In[15]:


arr[::-1]


# ## 间接排序：argsort和lexsort

# In[17]:


values = np.array([5, 0, 1, 3, 2])
indexer = values.argsort()
indexer


# In[18]:


values[indexer]


# In[19]:


arr = randn(3,5)
arr[0] = values
arr


# In[20]:


arr[:,arr[0].argsort()]


# In[23]:


first_name = np.array(['Bob', 'Jane', 'Steve', 'Bill', 'Barbara'])
last_name = np.array(['Jones', 'Arnold', 'Arnold', 'Jones', 'Walters'])
sorter = np.lexsort((first_name,last_name))
for i,j in zip(last_name[sorter],first_name[sorter]):
    print(i,j)


# In[24]:


values = np.array(['2:first', '2:second', '1:first', '1:second', '1:third'])
key = np.array([2,2,1,1,1])
indexer = key.argsort(kind = 'mergesort')
indexer


# In[25]:


values.take(indexer)


# ## np.searchsorted在有序数组中查找元素

# In[26]:


arr = np.array([0, 1, 7, 12, 15])
arr.searchsorted(9)


# In[27]:


arr.searchsorted([0,8,6,4])


# In[28]:


arr.searchsorted([0,1],side = 'right')


# In[29]:


data = np.floor(np.random.uniform(0, 10000, size=50))
bins = np.array([0,100,1000,5000,10000])
data


# In[31]:


labels = bins.searchsorted(data)
labels


# In[32]:


from pandas import Series
Series(data).groupby(labels).mean()


# In[33]:


np.digitize(data, bins)


# In[34]:


X =  np.array([[ 8.82768214,  3.82222409, -1.14276475,  2.04411587],
                     [ 3.82222409,  6.75272284,  0.83909108,  2.08293758],
                     [-1.14276475,  0.83909108,   5.01690521,  0.79573241],
                     [ 2.04411587,  2.08293758,  0.79573241,  6.24095859]])
X[:,0]


# In[35]:


y = X[:, :1]
y


# In[36]:


np.dot(y.T,np.dot(X,y))


# ## np.matrix

# In[37]:


Xm = np.matrix(X)
Xm


# In[39]:


ym = Xm[:,0]
ym


# In[40]:


ym.T*Xm*ym


# In[41]:


Xm.I


# ### 高级数组输入输出

# In[42]:


mmap = np.memmap('mymmap', dtype='float64', mode='w+', shape=(10000, 10000))
mmap


# In[43]:


section = mmap[:5]


# In[44]:


section[:] = np.random.randn(5, 10000)
mmap.flush()


# In[45]:


mmap


# In[46]:


arr_c = np.ones((1000,1000),order = 'C')
arr_F = np.ones((1000,1000),order = 'F')
arr_c.flags


# In[47]:


arr_F.flags


# In[50]:


get_ipython().run_line_magic('timeit', 'arr_c.sum(1)')


# In[51]:


get_ipython().run_line_magic('timeit', 'arr_F.sum(1)')

