
# coding: utf-8

# In[2]:


from pandas import DataFrame,Series
import pandas as pd


# In[4]:


obj = Series([1,2,3,4],index = ['a','b','c','d'])
obj


# In[5]:


obj[['a','b','c']]


# In[6]:


import numpy as np
np.exp(obj)


# In[8]:


obj


# In[10]:


sdata = {u'湖南':6800,u'陕西':3700,u'河南':10000,u'河北':5000}
obj2 = Series(sdata)
obj2


# In[12]:


provices = [u'湖南',u'陕西',u'河南',u'辽宁']
obj3 = Series(sdata,index = provices)
obj3


# In[16]:


obj3.notnull()
#pd.notna(obj3)


# In[17]:


obj2+obj3


# In[18]:


obj3.name = 'population'
obj3


# In[22]:


data = {'states':['Ohio','Ohio','Ohio','Texas','Texas'],
           'year':[2011,2012,2013,2011,2012],
           'pop':[1.1,1.3,1.4,2.4,2.3]}
frame = DataFrame(data,columns=['year','pop','states'],index=['壹','贰','叁','肆','伍'])
frame


# In[32]:


frame.loc['叁']


# In[37]:


frame['debt'] = np.random.randn(5)
frame


# In[38]:


del frame['debt']
frame


# In[39]:


pop = {'Nevada': {2001: 2.4, 2002: 2.9},
'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame2 = DataFrame(pop)
frame2


# In[42]:


frame2 = DataFrame(pop,index=[2001,2002,2000])
frame2


# In[43]:


2003 in frame2.index


# In[18]:


obj = Series([1.3,1.5,-7.8,5.6],index=['d','a','b','c'])
obj2 = obj.reindex(['a','b','v','c','d'])
obj2


# In[19]:


obj.reindex(['a','b','c','d','e'],fill_value=0)


# In[27]:


obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6),method='backfill')


# In[35]:


frame = DataFrame(np.arange(9).reshape((3,3)),
                index= ['a' , 'c' , 'b'],
                columns =['Ohio','Texas','Cali'])
frame


# In[36]:


frame2 = frame.reindex(['a','b','c','d'])
frame2


# In[45]:


states = ['Texas','Cali','Utah']
frame.reindex(columns=states)


# In[52]:


frame.loc[['a','b','c'],['Ohio','Cali']]


# In[55]:


obj = Series(np.arange(5.),index=['a','b','c','d','e'])
obj.drop(['c','d'])


# In[57]:


data = DataFrame(np.arange(16).reshape((4,4)),
                index=['Ohio', 'Colorado', 'Utah', 'New York'],
                columns=['one','two','three','four'])
data


# In[58]:


data.drop(['Ohio','Utah'])


# In[60]:


data.drop(['one'],axis=1)


# In[64]:


obj = Series(np.random.randn(8))
print(obj)
obj.sort_values()


# In[69]:


obj = Series(np.arange(4.),index=['a','b','c','d'])
obj


# In[66]:


obj[1]


# In[68]:


obj[obj>2]


# In[70]:


obj['a':'c']


# In[75]:


data.iloc[3]
#data.loc['Ohio']


# In[79]:


data[data['three']>4]['four']


# In[80]:


data<5


# In[84]:


s1 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
s2 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
s3 = s1+s2
s3


# In[89]:


df1 = DataFrame(np.arange(9.).reshape((3,3)),
               columns=list('bcd'),
               index=['Ohio', 'Texas', 'Colorado'])
df2 =  DataFrame(np.arange(12.).reshape((4, 3)), 
                 columns=list('bde'),
                 index=['Utah', 'Ohio', 'Texas', 'Oregon'])
df1+df2


# In[99]:


df1 = DataFrame(np.arange(12.).reshape((3,4)),
               columns=list('abcd'))
df2 = DataFrame(np.arange(16.).reshape((4,4)),
               columns=list('bcde'))
df2.add(df1,fill_value=0)


# In[100]:


data


# In[101]:


series = data.iloc[0]


# In[102]:


data-series


# In[103]:


series2 = data['two']
series2


# In[104]:


data.sub(series2,axis=0)


# In[108]:


frame = DataFrame(np.random.randn(4, 3), columns=list('bde'),
                  index=['Utah', 'Ohio', 'Texas', 'Oregon'])
np.abs(frame)
frame


# In[109]:


f = lambda x: x.max() - x.min()
frame.apply(f,axis=1)


# In[118]:


frame = DataFrame(np.arange(8).reshape((2, 4)), index=['three', 'one'],
                  columns=['d', 'a', 'b', 'c'])
print(frame)
frame.sort_index(ascending=False)


# In[123]:


frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame.sort_values(by=['a','b'])


# In[126]:


obj = Series([7, -5, 7, 4, 2, 0, 4])
obj.rank(method='average')


# In[128]:


obj = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
obj.index.is_unique


# In[3]:


frame = DataFrame({'b': [4, 7, -3, 2], 'a': [0, 1, 0, 1]})
frame.sort_values(by = 'b')


# ### 汇总和计算描述统计

# In[4]:


import numpy as np
df = DataFrame([[1.4,np.nan],[7.1,4.5],[np.nan,np.nan],[0.75,-1.3]],
                index = ['a','b','c','d'],
                columns = ['one','two'])
df


# In[8]:


df.sum()
df.sum(axis = 1) # axis = 1 指的是columns方向
df.sum(axis = 1,skipna = False)


# In[9]:


df.mean(axis = 1)


# In[12]:


df.cumsum(axis = 1)


# In[14]:


df.describe()


# In[16]:


df.quantile(0.25)


# In[20]:


import pandas.io.data as web
all_data = {}
for ticker in ['AAPL', 'IBM', 'MSFT', 'GOOG']:
   all_data[ticker] = web.get_data_yahoo(ticker, '1/1/2000', '1/1/2010')
price = DataFrame({tic: data['Adj Close']
                       for tic, data in all_data.iteritems()})
volume = DataFrame({tic: data['Volume']
                       for tic, data in all_data.iteritems()})
returns = price.pct_change()
returns


# ## 唯一值、值计数以及成员资格

# In[21]:


obj = Series(['c', 'a', 'd', 'a', 'a', 'b', 'b', 'c', 'c'])
unique = obj.unique()
unique


# In[25]:


obj.value_counts()


# In[26]:


msk = obj.isin(['a','c'])
msk


# In[27]:


obj[msk]


# In[42]:


data = DataFrame({'Qu1': [1, 3, 4, 3, 4],
                  'Qu2': [2, 3, 1, 2, 3],
                  'Qu3': [1, 5, 2, 4, 4]})
print(data)
data.plot(kind='bar')


# In[45]:


result = data.apply(pd.value_counts).fillna(0)
print(result)
result.plot(kind = 'bar')


# ## 处理缺失数据

# In[46]:


string_data = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
string_data.isnull()


# In[47]:


string_data[0] = None
string_data.isnull()


# In[49]:


s_data = string_data.dropna()
s_data


# In[53]:


from numpy import nan as NA
data = DataFrame([[1., 6.5, 3.], [1., NA, NA],
                        [NA, NA, NA], [NA, 6.5, 3.]])
data


# In[54]:


data.dropna(how='all')


# In[59]:


data.fillna(data.mean())


# ## 层次化索引

# In[9]:


import numpy as np
data = Series(np.random.randn(10),index = [['a','a','a','b','b','b'
                                           ,'c','c','d','d'],
                                          [1,2,3,1,2,3,1,2,2,3]])
data


# In[10]:


data.index


# In[11]:


data['b']


# In[12]:


data['b':'c']


# In[13]:


data.loc[['b','d']]


# In[18]:


data[:,2]


# In[20]:


df = data.unstack()
df


# In[21]:


frame = DataFrame(np.arange(12).reshape((4, 3)),
                  index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
                  columns=[['Ohio', 'Ohio', 'Colorado'],
                           ['Green', 'Red', 'Green']])
frame


# In[24]:


frame.columns


# In[25]:


frame.index.names = ['key1','key2']
frame.columns.names = ['states','color']
frame


# In[28]:


frame.swaplevel('key1','key2')


# In[36]:


frame.sort_index(level=0)


# ## 根据级别汇总成列

# In[37]:


frame.sum(level = 1)


# In[40]:


frame.sum(level='color',axis=1) # 设置axis为1 才认为color为level


# In[41]:


frame = DataFrame({'a': range(7), 'b': range(7, 0, -1),
                   'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'],
                   'd': [0, 1, 2, 0, 1, 2, 3]})
frame


# In[42]:


frame.set_index(['c','d'])


# In[43]:


frame.reset_index()


# ## 整数索引

# In[45]:


ser = Series(np.arange(3.))
ser


# In[49]:


ser.iloc[-1]

