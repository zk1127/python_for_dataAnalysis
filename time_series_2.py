
# coding: utf-8

# In[1]:


from datetime import datetime
from pandas import DataFrame,Series
import numpy as np
import pandas as pd


# In[2]:


import pytz
pytz.common_timezones[60:65]


# In[3]:


pytz.timezone('US/Eastern')


# In[4]:


rng = pd.date_range('1/2/2011 9:30',periods=6,freq='D')
ts = Series(np.random.randn(6),index = rng)
ts


# In[5]:


print(ts.index.tz)


# In[6]:


pd.date_range('1/2/2011 9:30',periods=6,freq='D',tz = 'UTC')


# In[7]:


ts_utc = ts.tz_localize('UTC')


# In[8]:


ts_utc


# In[9]:


ts_utc.tz_convert('US/Eastern')


# In[10]:


ts_eastern = ts.tz_localize('US/Eastern')
ts_eastern.tz_convert('Europe/Berlin')


# ### 操作时区意识型TimeStamp对象

# In[11]:


stamp = pd.Timestamp('2018-3-2 04:00')
stamp_utc = stamp.tz_localize('UTC')
stamp_utc.tz_convert('Asia/Shanghai')
# stamp_utc


# In[12]:


stamp_moscow = pd.Timestamp('2018-4-5 9:00',tz = 'Europe/Moscow')
stamp_moscow


# In[13]:


now = datetime.now()
stamp_utc.value


# In[14]:


rng = pd.date_range('3-18-2018',periods = 6,freq = 'B')
ts = Series(np.random.rand(6),index = rng)


# In[15]:


ts


# In[16]:


ts1 = ts[:3].tz_localize('Asia/Shanghai')
ts2 = ts[4:].tz_localize('US/Eastern')
ts_re = ts1 + ts2
ts_re


# In[17]:


ts_re.index


# ### 时期及其算术运算

# In[18]:


p = pd.Period(2007,freq = 'A-DEC')
p


# In[19]:


pd.Period('2014', freq='A-DEC') - p


# In[21]:


rng = pd.date_range('2-1-2013','3-2-2015',freq = 'MS')
rng


# In[23]:


Series(np.random.rand(26),index = rng)


# In[25]:


values = ['2001Q3', '2002Q2', '2003Q1']
index = pd.PeriodIndex(values ,freq = 'Q-DEC')
index


# In[27]:


p = pd.Period('2007',freq = 'A-DEC')
p.asfreq('M',how = 'end')


# In[28]:


p = pd.Period('2007-08', 'M')
p.asfreq('A-JUN')


# In[35]:


rng = pd.date_range('2007','2010',freq = 'A-DEC')
ts = Series(np.random.randn(len(rng)),index = rng)
ts


# In[36]:


ts.asfreq('M', how='start')


# In[37]:


p = pd.Period('2012Q4', freq='Q-JAN')
p


# In[38]:


p.asfreq('D', 'start')


# In[39]:


p4pm = (p.asfreq('B', 'e') - 1).asfreq('T', 's') + 16 * 60
p4pm


# In[40]:


p4pm.to_timestamp()


# In[41]:


rng = pd.date_range('1/29/2000', periods=6, freq='D')
ts2 = Series(np.random.randn(6),index = rng)
ts2.to_period('M')


# In[42]:


macrodata_file = open('F:/电影/数据分析/pydata-book-master/ch08/macrodata.csv')
data = pd.read_csv(macrodata_file)


# In[44]:


data.head()


# In[45]:


index = pd.PeriodIndex(year=data.year, quarter=data.quarter, freq='Q-DEC')
data.index = index


# In[46]:


data.head()


# In[48]:


data.infl.plot()


# ### 重采样和频率转换

# In[2]:


rng = pd.date_range('3/4/2018',periods=100,freq = 'D')
ts = Series(np.random.randn(100),index = rng)
ts.head()


# In[5]:


ts.resample( 'M' ).mean()


# In[7]:


# ts.resample('M',kind = 'period').mean()
ts.resample('M',kind = 'timestamp').mean()


# In[13]:


rng = pd.date_range('1/1/2000', periods=12, freq='T')
ts = Series(np.arange(12),index = rng)
ts


# In[17]:


ts.resample('5min',closed='right').sum()


# In[18]:


ts.resample('5min',closed='left').sum()


# In[19]:


ts.resample('5min',closed='left',label = 'right').sum() # closed控制闭区间的位置，label控制结果的区间的位置


# In[20]:


ts.resample('5min',closed ='right',label = 'right',loffset='-5s').sum()


# In[24]:


# 金融领域开盘收盘最高最低
ts.resample('5min',closed = 'right',label = 'right').ohlc()


# In[25]:


rng = pd.date_range('1/1/2000', periods=100, freq='D')
ts = Series(np.arange(100), index=rng)
ts.groupby(lambda x: x.month).mean()


# In[26]:


ts.groupby(lambda x : x.weekday).mean()


# In[30]:


# 升采样
frame = DataFrame(np.random.randn(2, 4),
                       index=pd.date_range('1/1/2000', periods=2, freq='W-WED'),
                       columns=['Colorado', 'Texas', 'New York', 'Ohio'])
df_daily = frame.resample('D')
df_daily.mean()


# In[32]:


frame.resample('D').ffill()


# In[33]:


frame = DataFrame(np.random.randn(24, 4),
                       index=pd.period_range('1-2000', '12-2001', freq='M'),
                       columns=['Colorado', 'Texas', 'New York', 'Ohio'])
frame[:4]


# In[36]:


anual_frame = frame.resample('A-DEC').ohlc()
anual_frame


# In[39]:


anual_frame.resample('Q-DEC').ffill()


# ## pandas 时间序列函数

# In[40]:


close_px_file = open('F:/电影/数据分析/pydata-book-master/ch09/stock_px.csv')
close_px_all = pd.read_csv(close_px_file,parse_dates=True,index_col=0)
close_px = close_px_all[['AAPL','MSFT','XOM']]
close_px.resample('B').ffill()
close_px[:6]


# In[42]:


close_px['AAPL'].plot()


# In[43]:


close_px.loc['2009'].plot()


# In[47]:


close_px['AAPL'].loc['2011-1':'2011-3'].plot(fontsize = 'larger')


# In[49]:


aapl_q = close_px['AAPL'].resample('Q-DEC').ffill()


# In[51]:


aapl_q.loc['2009':].plot()


# ### 移动窗口函数

# In[2]:


close_px_file = open('F:/电影/数据分析/pydata-book-master/ch09/stock_px.csv')
close_px_all = pd.read_csv(close_px_file,parse_dates=True,index_col=0)
close_px = close_px_all[['AAPL','MSFT','XOM']]
close_px.resample('B').ffill()
close_px[:6]


# In[7]:


pd.rolling_mean(close_px.AAPL,250).plot()


# In[10]:


appl_std250 = close_px['AAPL'].rolling(250, min_periods=10).std()


# In[11]:


appl_std250[7:12]


# In[12]:


appl_std250.plot()


# In[16]:


close_px.rolling(250,min_periods=60).mean().plot()


# ### 指数加权函数

# In[23]:


import matplotlib.pyplot as plt
fig,axes = plt.subplots(nrows = 2,ncols = 1,sharex = True,sharey = True,figsize = (12,7))
aapl_px = close_px.AAPL.loc['2005':'2009']
ma60 = aapl_px.rolling(60,min_periods = 50).mean()
ewma60 =aapl_px.ewm(span = 60).mean()
aapl_px.plot(style = 'k-',ax = axes[0])
ma60.plot(style = 'k--',ax = axes[0])
aapl_px.plot(style = 'k-',ax = axes[1])
ewma60.plot(style = 'k--',ax = axes[1])
axes[0].set_title('Simple MA')
axes[1].set_title('Exponentially-weighted MA')


# In[28]:


spx = close_px_all['SPX']
spx_pct = spx / spx.shift(1) -1
returns = close_px.pct_change()
corr = returns.AAPL.rolling(125,min_periods=100).corr(other = spx_pct)


# In[29]:


corr.plot()


# In[30]:


returns.rolling(125,min_periods=100).corr(other = spx_pct).plot()


# In[31]:


from scipy.stats import percentileofscore
score_at_2percent = lambda x :percentileofscore(x,0.02)
result = returns.AAPL.rolling(250).apply(score_at_2percent)
result.plot()

