
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import DataFrame,Series
import numpy as np
df = DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],
            'key2' : ['one', 'two', 'one', 'two', 'one'],
            'data1' : np.random.randn(5),
            'data2' : np.random.randn(5)})
df


# In[2]:


tips_file = open('F:/电影/数据分析/pydata-book-master/ch08/tips.csv')
tips = pd.read_csv(tips_file)
tips['tip_pct'] = tips.tip / tips.total_bill


# In[3]:


def top(df,n = 5,column = 'tip_pct'):
    return df.sort_values(by = column)[-n:]


# In[4]:


top(tips,n = 6)


# In[5]:


tips.groupby('smoker').apply(top)


# In[6]:


tips.groupby(['smoker','day']).apply(top,n = 5,column = 'total_bill')


# ### 禁止分组键

# In[7]:


tips.groupby('smoker',group_keys=False).apply(top)


# In[8]:


frame = DataFrame({'data1': np.random.randn(1000),
                   'data2': np.random.randn(1000)})
factor = pd.cut(frame.data1, 4)
factor[:4]


# In[9]:


def get_stats(group):
        return {'min': group.min(), 'max': group.max(),
                'count': group.count(), 'mean': group.mean()}
grouped = frame.data2.groupby(factor)
grouped.apply(get_stats).unstack()


# In[10]:


s = Series(np.random.randn(6))
s[::5] = np.nan


# In[11]:


s


# In[12]:


s.fillna(s.mean())


# In[13]:


# Hearts（红桃）, Spades（黑桃）, Clubs（梅花）, Diamonds（方片）
suits = ['H', 'S', 'C', 'D']
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ['A'] + list(range(2, 11)) + ['J', 'K', 'Q']
cards = []
for suit in ['H', 'S', 'C', 'D']:
    cards.extend(str(num) + suit for num in base_names)

deck = Series(card_val, index=cards)
deck


# In[14]:


def draw(deck,n = 5):
    return deck.take(np.random.permutation(len(deck)))[:n]
draw(deck)


# In[15]:


get_suit = lambda card : card[:1]
deck.groupby(get_suit).apply(draw,n=2)


# In[16]:


df = DataFrame({'category': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
      'data': np.random.randn(8),
      'weights': np.random.rand(8)})
df


# In[17]:


grouped = df.groupby('category')
get_wavg = lambda g : np.average(g['data'],weights=g['weights'])
grouped.apply(get_wavg)


# In[18]:


close_px_file = open('F:/电影/数据分析/pydata-book-master/ch09/stock_px.csv')
close_px = pd.read_csv(close_px_file,parse_dates=True,index_col=0)
close_px[:3]


# In[19]:


close_px[['AAPL','MSFT','XOM']].plot()


# In[20]:


rets = close_px.pct_change().dropna()
spx_corr = lambda x : x.corrwith(x['SPX'])
by_year = rets.groupby(lambda x : x.year)
year_corr = by_year.apply(spx_corr)
year_corr


# In[21]:


year_corr.plot()


# In[22]:


by_year.apply(lambda x : x['AAPL'].corr(x['MSFT']))


# In[23]:


import statsmodels.api as sm
def regress(data,yvar,xvar):
    Y = data[yvar]
    X = data[xvar]
    X['intercept'] = 1
    result = sm.OLS(Y,X).fit()
    return result.params


# In[24]:


by_year.apply(regress, 'AAPL', ['SPX'])


# In[25]:


tips[:4]


# In[26]:


tips.pivot_table(index = ['sex','smoker'])


# In[27]:


tips.pivot_table(['tip_pct','size'],index = ['sex','day'],columns = 'smoker')


# In[28]:


tips.pivot_table(['tip_pct','size'],index = ['sex','day'],columns = 'smoker',margins=True)


# In[31]:


tips.pivot_table('tip_pct',index = ['sex','day'],columns = 'smoker',aggfunc=len,margins=True).plot()


# In[30]:


pd.crosstab([tips.sex, tips.day], tips.smoker, margins=True)


# In[32]:


fec_file = open('F:/电影/数据分析/pydata-book-master/ch09/P00000001-ALL.csv')
fec = pd.read_csv(fec_file)
fec[:3]


# In[33]:


fec.iloc[12345]


# In[34]:


parties = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}


# In[35]:


fec.cand_nm[123456:123461].map(parties)


# In[36]:


fec['party'] = fec.cand_nm.map(parties)


# In[37]:


fec['party'].value_counts()


# In[38]:


fec = fec[fec.contb_receipt_amt > 0]


# In[39]:


fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack', 'Romney, Mitt'])]


# In[40]:


fec.contbr_occupation.value_counts()[:3]


# In[41]:


occ_mapping = {
    'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
    'INFORMATION REQUESTED' : 'NOT PROVIDED',
    'INFORMATION REQUESTED (BEST EFFORTS)' : 'NOT PROVIDED',
    'C.E.O.': 'CEO'
}

# 如果没有提供相关映射，则返回x
f = lambda x: occ_mapping.get(x, x)
fec.contbr_occupation = fec.contbr_occupation.map(f)


# In[42]:


emp_mapping = {
    'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
    'INFORMATION REQUESTED' : 'NOT PROVIDED',
    'SELF' : 'SELF-EMPLOYED',
    'SELF EMPLOYED' : 'SELF-EMPLOYED',
}

# 如果没有提供相关映射，则返回x
f = lambda x: emp_mapping.get(x, x)
fec.contbr_employer = fec.contbr_employer.map(f)


# In[44]:


by_occupation = fec.pivot_table('contb_receipt_amt',
                                index='contbr_occupation',
                                columns='party', aggfunc='sum')


# In[45]:


over_2mm = by_occupation[by_occupation.sum(1) > 2000000]
over_2mm.plot(kind = 'barh')


# In[51]:


def get_top_amounts(group, key, n=5):
    totals = group.groupby(key)['contb_receipt_amt'].sum()

    # 根据key对totals进行降序排列
    return totals.sort_values(ascending=False)[n:]


# In[52]:


grouped = fec_mrbo.groupby('cand_nm')
grouped.apply(get_top_amounts, 'contbr_occupation', n=7)


# In[54]:


bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
labels = pd.cut(fec_mrbo.contb_receipt_amt, bins)
labels


# In[55]:


grouped = fec_mrbo.groupby(['cand_nm', labels])
grouped.size().unstack(0)


# In[56]:


bucket_sums = grouped.contb_receipt_amt.sum().unstack(0)
normed_sums = bucket_sums.div(bucket_sums.sum(axis=1), axis=0)


# In[57]:


normed_sums.plot(kind = 'barh',stacked = True)

