
# coding: utf-8

# In[1]:


import json
path = u'F:/电影/练数成金/pydata-book-master/ch02/usagov_bitly_data2012-03-16-1331923249.txt'
records = [json.loads(line) for line in open(path)]


# In[2]:


records[0]


# In[3]:


from collections import defaultdict

def get_count(sequene):
    counts = defaultdict(int)
    for x in sequene:
        counts[x] += 1
    return counts

time_zones = [rec['tz'] for rec in records if 'tz' in rec]


# In[4]:


tz_count = get_count(time_zones)


# In[5]:


tz_count['America/New_York']


# In[6]:


from collections import Counter
tz_count = Counter(time_zones)
tz_count.most_common(10)


# In[7]:


from pandas import DataFrame,Series
import pandas as pd
import numpy as np


# In[8]:


frame = DataFrame(records)
frame['tz'][:10]


# In[9]:


tz_count = frame['tz'].value_counts()


# In[10]:


tz_count[:10]


# In[11]:


clearn_tz = frame['tz'].fillna('Missing')
clearn_tz[clearn_tz == ''] = 'Unknown'
tz = clearn_tz.value_counts()
tz[:10]


# In[13]:


tz[:10].plot(kind = 'barh')


# In[14]:


frame['a'][:10]


# In[15]:


browers = Series(x.split()[0] for x in frame['a'].dropna())
browers[:10]


# In[16]:


cframe = frame[frame.a.notnull()]
opreating_system = np.where(cframe['a'].str.contains('Windows'),'Windows','Not Windows')
opreating_system[:5]


# In[17]:


by_tz_os = cframe.groupby(['tz',opreating_system])
agg_counts = by_tz_os.size().unstack().fillna(0)
agg_counts[:5]


# In[18]:


indexer = agg_counts.sum(1).argsort()
indexer[:10]


# In[19]:


count_subset = agg_counts.take(indexer)[-10:]


# In[20]:


count_subset


# In[21]:


count_subset.plot(kind='barh',stacked = True)


# In[22]:


normed_subset = count_subset.div(count_subset.sum(1),axis=0)
normed_subset.plot(kind='barh',stacked = True)

