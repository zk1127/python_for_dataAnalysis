
# coding: utf-8

# In[1]:


import pandas as pd
years = range(1880,2011)
pieces = []
columns = ['name','sex','births']
for year in years:
    path = open(u'F:/电影/练数成金/pydata-book-master/ch02/names/yob{0}.txt'.format(year))
    frame = pd.read_csv(path,names = columns)
    
    frame['year'] = year
    pieces.append(frame)
names = pd.concat(pieces,ignore_index=True)
names[:10]


# In[2]:


total_births = names.pivot_table('births',index='year',columns='sex',aggfunc=sum)
total_births.tail()


# In[3]:


total_births.plot(title='Total birth by sex and year')


# In[4]:


def add_prop(group):
    births = group.births
    group['prop'] = births/births.sum()
    return group
names = names.groupby(['year','sex']).apply(add_prop)
names


# In[5]:


import numpy as np
np.allclose( names.groupby(['year','sex']).prop.sum(),1)


# In[5]:


pieces = []
for year,group in names.groupby(['year','sex']):
    pieces.append(group.sort_values(by='births', ascending=False)[:1000])
top1000 = pd.concat(pieces,ignore_index=True)


# In[13]:


top1000


# In[6]:


boys = top1000[top1000.sex =='M']
girls = top1000[top1000.sex == 'F']
total_births = top1000.pivot_table('births',index='year',columns='name',aggfunc=sum)
total_births


# In[7]:


subset =total_births[['John','Harry','Mary','Gilbert','Paul']]
subset.plot(subplots=True,figsize = (12,10), grid=False,title="Number of births per year")


# In[27]:


subset.dtypes


# In[8]:


table = top1000.pivot_table('prop',index = 'year',columns='sex',aggfunc=sum)
table.plot(title = 'Sum of top 1000 prop by sex and year')


# In[10]:


df = boys[boys.year == 2010]
prop_cumsum = df.sort_values(by='prop',ascending = False).prop.cumsum()
prop_cumsum[:10]


# In[11]:


prop_cumsum.searchsorted(0.5)


# In[27]:


def get_quantile_count(group,q=0.5):
    group = group.sort_values(by='prop',ascending = False)
    return group.prop.cumsum().searchsorted(q)[0]+1

diversity = top1000.groupby(['year','sex']).apply(get_quantile_count)
diversity = diversity.unstack('sex')
diversity.head()
diversity.plot(title = 'Nunber of popular names in top 50%')


# In[28]:


get_last_letter = lambda x : x[-1]
last_letters = names.name.map(get_last_letter)
last_letters.name = 'last_letter'
table = names.pivot_table('births',index = last_letters,columns=['sex','year'],aggfunc=sum)
subtable = table.reindex(columns=[1910,1960,2010],level='year')
subtable.head()


# In[31]:


letter_prop = subtable / subtable.sum().astype(float)
import matplotlib.pyplot as plt
fig,axes = plt.subplots(2,1,figsize = (12,10))
letter_prop['M'].plot(kind='bar',rot = 0,ax = axes[0],title = 'Male')
letter_prop['F'].plot(kind='bar',rot = 0,ax = axes[1],title = 'Female')


# In[35]:


letter_prop = table/table.sum().astype(float)
dny_ts = letter_prop.loc[['d','n','y'],'M'].T
dny_ts[:10]


# In[36]:


dny_ts.plot()


# In[39]:


import numpy as np
all_names = names.name.unique()
mask = np.array(['lesl' in x.lower() for x in all_names])
lesley_like = all_names[mask]
lesley_like


# In[40]:


fitlered = top1000[top1000.name.isin(lesley_like)]
fitlered.groupby('name').births.sum()


# In[42]:


table = fitlered.pivot_table('births',index = 'year',columns='sex',aggfunc=sum)
table = table.div(table.sum(1),axis=0)
table.tail()


# In[45]:


table.plot(style={'M':'k-','F':'k--'})

