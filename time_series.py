
# coding: utf-8

# ## Python日期时间

# In[1]:


from datetime import datetime


# In[2]:


now = datetime.now()


# In[3]:


now


# In[4]:


time = datetime.strptime('2017/01/11 12:23:45','%Y/%m/%d %H:%M:%S')
time


# In[5]:


from datetime import timedelta
start = datetime(2017,2,23)
start = start +timedelta(12)
start


# In[6]:


stime = datetime(2011,7,8)
stime.strftime('%Y-%m-%d')


# In[7]:


from dateutil.parser import parse
parse('Jan 30, 1778 1:45 PM')


# In[8]:


parse('1/2/2067',dayfirst = True)


# ### 注：Python数据分析书的百分之68部分有datetime格式定义表

# ### 时间序列基础

# In[11]:


from datetime import datetime
from pandas import Series,DataFrame
import numpy as np
dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7),datetime(2011, 1, 8), datetime(2011, 1, 10), datetime(2011, 1, 12)]
ts = Series(np.random.randn(6),index =dates)
ts


# In[12]:


type(ts)


# In[13]:


ts.index


# In[16]:


ts[::2]


# In[18]:


stamp = ts.index[0]
stamp


# In[20]:


ts['1/2/2011']


# In[22]:


import pandas as pd
long_ts = Series(np.random.randn(1000),index = pd.date_range('1/1/2011',periods = 1000))
long_ts[:10]


# In[24]:


long_ts['2013'][:10]


# In[25]:


long_ts['2012/05']


# In[29]:


ts[datetime(2011,1,7):datetime(2011,1,12)]


# In[30]:


ts[datetime(2011,1,6):datetime(2011,1,9)]


# In[31]:


ts.truncate(after='1/7/2011')


# In[33]:


dates = pd.date_range('1/1/2000', periods=100, freq='W-WED')
long_df = DataFrame(np.random.randn(100, 4),
                          index=dates,
                         columns=['Colorado', 'Texas', 'New York', 'Ohio'])
long_df.loc['5-2001']


# In[34]:


dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', '1/2/2000', '1/2/2000','1/3/2000'])
dup_ts = Series(np.random.randn(5),index = dates)
dup_ts


# In[35]:


dup_ts['2000-1-2']


# In[36]:


dup_ts.groupby(level = 0).mean()


# ### 日期的范围、频率以及移动

# In[37]:


ts


# In[42]:


ts_sample = ts.resample('D')
ts_sample.count()


# In[47]:


index = pd.date_range('4/1/2013','6/1/2013')
index


# In[50]:


pd.date_range(start='4/1/2012', periods=20,freq = 'D')


# In[54]:


pd.date_range('1/1/2015','12/1/2015',freq = 'BM')


# In[56]:


from pandas.tseries.offsets import Hour,Minute
hour = Hour(4)
hour


# In[57]:


pd.date_range('1/1/2000', '1/3/2000 23:59', freq='4h')


# In[58]:


pd.date_range('1/1/2000',periods=10,freq='1h30min')


# In[60]:


rng = pd.date_range('1/1/2012','9/1/2012',freq = 'WOM-3Fri')
list(rng)


# In[63]:


ts = Series(np.random.randn(4),
            index=pd.date_range('1/1/2000', periods=4, freq='M'))
ts


# In[62]:


ts.shift(2)


# In[69]:


'{}%'.format((ts/ts.shift(1) - 1).iloc[2]*100)


# In[70]:


ts.shift(2,freq = 'M')


# In[71]:


from pandas.tseries.offsets import Day,MonthEnd
now = datetime.now()


# In[72]:


now + MonthEnd()


# In[73]:


now + MonthEnd(2)

