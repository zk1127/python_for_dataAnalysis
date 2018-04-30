
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


data1 = [[1,2,3,4],[5,4,6,7]]
array1 = np.array(data1)


# In[3]:


print(array1.shape)
array1


# In[4]:


np.zeros(10)


# In[5]:


np.zeros((3,5))


# In[9]:


type(array1.shape)


# In[10]:


np.empty((2,3,5))


# In[11]:


np.arange(23)


# In[12]:


np.eye(3)


# In[15]:


arr1 = np.array([1,2,3],dtype=np.float64)
arr1


# In[17]:


arr2 = np.array([1,2,3,4,5])
float_arr2 = arr2.astype(np.float64)
float_arr2


# In[19]:


arr3 = np.array([[1.,2.,3.],[4.,5.,6.]])
arr3*arr3


# In[21]:


arr3 * arr3 - arr3


# In[22]:


1/arr3


# In[23]:


arr4 = np.arange(10)
arr4[5:9]


# In[24]:


arr4[3:5] = 8
arr4


# In[25]:


arr4_slice = arr4[5:8].copy()
arr4_slice[:] = 123
arr4


# In[27]:


arr2d = np.array([[1,2,3],[2,7,3],[7,8,9]])
arr2d


# In[28]:


arr2d[2]


# In[29]:


arr2d[0][2]


# In[30]:


arr2d[:3]


# In[32]:


arr2d[:,:2]


# In[33]:


arr2d[:,2]


# In[34]:


arr2d[:2,1:]


# In[2]:


from numpy.random import randn
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.array(randn(7,4))
data


# In[44]:


names == 'Bob'


# In[45]:


data[names == 'Bob']


# In[46]:


mask = (names == 'Bob')|(names == 'Joe')
data[mask,2:]


# In[48]:


arr5 = np.arange(32).reshape((8,4))
arr5


# In[49]:


arr5[[1, 5, 7, 2], [0, 3, 1, 2]] # 构成索引对（1，0），（5，3）.。。。


# In[51]:


arr6 = np.random.randn(4,5)
np.dot(arr6.T,arr6)


# In[53]:


arr7 = np.arange(10)
np.sqrt(arr7)


# In[54]:


np.exp(arr7)


# In[55]:


np.max(arr7)


# In[57]:


np.maximum(arr7,arr7.T)


# In[58]:


np.sign(arr7)


# In[63]:


points = np.arange(-5,5,0.01)
xs,ys = np.meshgrid(points,points)
z = np.sqrt(xs**2+ys**2)
import matplotlib.pyplot as plt
plt.imshow(z,cmap = plt.cm.gray)
plt.colorbar()


# In[64]:


xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
arr8 = np.where(cond,xarr,yarr)
arr8


# In[65]:


arr9 = np.random.randn(4,5)
arr9.mean()


# In[66]:


arr9.sum(axis = 1)


# In[67]:


print(arr9)
arr9.cumsum(0)


# In[68]:


arr9.cumprod()


# In[70]:


arr9.cumprod(1)


# In[76]:


arr10 = randn(10)
arr10


# In[77]:


arr10.sort()
arr10


# In[78]:


np.unique(names)


# In[79]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x,y)


# In[81]:


np.dot(x,np.ones(3))


# In[83]:


from numpy.linalg import inv,qr
X = randn(5,5)
mat = X.T.dot(X)
inv(mat)


# In[84]:


q,r = qr(mat)
q,r


# In[89]:


sample = np.random.randn(5,5)


# In[90]:


from random import normalvariate
N = 1000000
get_ipython().run_line_magic('timeit', 'sample = [normalvariate(0,1) for _ in range(N)]')


# In[91]:


get_ipython().run_line_magic('timeit', 'sample = np.random.normal(size = N)')


# In[6]:


import random
import matplotlib.pyplot as plt
position = 0
walk = [position]
for i in range(1000):
    step = 1 if random.randint(0,1) else -1
    position += step
    walk.append(position)
plt.plot(range(1001),walk,label ="second line")
plt.legend()
plt.show()

