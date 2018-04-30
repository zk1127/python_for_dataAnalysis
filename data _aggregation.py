
# coding: utf-8

# ## GroupBy 技术 - 拆分应用合并

# In[4]:


import pandas as pd
from pandas import DataFrame,Series
import numpy as np
df = DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
            'key2' : ['one', 'two', 'one', 'two', 'one'],
            'data1' : np.random.randn(5),
            'data2' : np.random.randn(5)})
df


# In[9]:


grouped = df['data1'].groupby(df['key1'])
grouped


# In[10]:


grouped.mean()


# In[12]:


means = df['data1'].groupby([df['key1'],df['key2']]).mean()
means


# In[13]:


means.unstack()


# In[15]:


states = np.array(['Ohio', 'California', 'California', 'Ohio', 'Ohio'])
years = np.array([2005, 2005, 2006, 2005, 2006])
df['data1'].groupby([states,years]).mean()


# In[16]:


df.groupby(['key1','key2']).mean()


# In[20]:


df.groupby(['key1','key2']).size()


# ### 对分组进行迭代

# In[23]:


for name,group in df.groupby(df['key1']):
    print(name)
    print(group)


# In[25]:


for (k1,k2),group in df.groupby([df['key1'],df['key2']]):
    print (k1,k2)
    print(group)


# In[31]:


pieces = dict(list(df.groupby(df['key1'])))
pieces['b']


# In[32]:


grouped = df.groupby(df.dtypes,axis = 1)
dict(list(grouped))


# ### 选取一个组或一个列

# In[33]:


df.groupby('key1')['data1']


# In[34]:


df.groupby('key1')[['data1']] 


# In[42]:


df.groupby(['key1','key2'])[['data2']].mean()


# In[43]:


s_group = df.groupby(['key1','key2'])['data2']
s_group.mean()


# In[44]:


people = DataFrame(np.random.randn(5, 5),
                   columns=['a', 'b', 'c', 'd', 'e'],
                   index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
people.ix[2:3,['b','c']] = np.nan
people


# In[45]:


mapping = {'a': 'red', 'b': 'red', 'c': 'blue', 'd': 'blue', 'e': 'red', 'f' : 'orange'}
by_columns = people.groupby(mapping,axis = 1)
by_columns.sum()


# In[46]:


map_series = Series(mapping)
people.groupby(map_series,axis = 1).count()


# In[47]:


people.groupby(len).sum()


# In[48]:


key_list = ['one', 'one', 'one', 'two', 'two']
people.groupby([len,key_list]).min()


# In[49]:


columns = pd.MultiIndex.from_arrays([['US', 'US', 'US', 'JP', 'JP'], [1, 3, 5, 1, 3]], names=['cty', 'tenor'])
hier_df = DataFrame(np.random.randn(4,5),columns=columns)
hier_df


# In[50]:


hier_df.groupby(level ='cty',axis =1).count()


# ## 数据聚合

# In[51]:


df


# In[52]:


grouped = df.groupby('key1')


# In[53]:


grouped['data1'].quantile(0.9)


# In[54]:


def peak_to_peak(arr):
    return arr.max() - arr.min()
grouped['data1'].agg(peak_to_peak)


# In[55]:


grouped.describe()


# In[56]:


tips_file = open('F:/电影/数据分析/pydata-book-master/ch08/tips.csv')
tips = pd.read_csv(tips_file)
tips[:3]


# In[57]:


tips['tip_pct'] = tips.tip / tips.total_bill


# In[58]:


tips[:3]


# In[59]:


grouped = tips.groupby(['sex','smoker'])


# In[62]:


grouped_pct = grouped['tip_pct'].mean()
grouped_pct


# In[65]:


def peak_to_peak(arr):
    return arr.max() - arr.min()
grouped_pct.agg(['mean','std',peak_to_peak])


# In[67]:


functions = ['count','mean','max']
result = grouped['tip_pct','tip'].agg(functions)


# In[68]:


result


# In[70]:


type(result['tip'])


# In[71]:


ftuples = [('Durchschnitt', 'mean'), ('Abweichung', np.var)]
grouped['tip_pct', 'total_bill'].agg(ftuples)


# In[72]:


grouped.agg({'tip':np.max,'size':'sum'})


# In[73]:


grouped.agg({'tip':['min','max','mean','std'],'size':'mean'})


# In[75]:


tips.groupby(['sex','smoker']).mean().reset_index()


# ### 分组级运算与转换

# In[76]:


df


# In[77]:


k1_means = df.groupby('key1').mean().add_prefix('mean_')
k1_means


# In[81]:


pd.merge(df,k1_means,left_on = 'key1',right_index=True)


# In[82]:


key = ['one', 'two', 'one', 'two', 'one']
people.groupby(key).mean()


# In[83]:


people.groupby(key).transform(np.mean)


# #### apply"拆分-应用-合并"
