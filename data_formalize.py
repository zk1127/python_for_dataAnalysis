
# coding: utf-8

# # 数据规整化，清理转换合并重塑

# ## 合并数据集

# In[1]:


import pandas as pd
from pandas import Series,DataFrame


# In[2]:


df1 = DataFrame({'key':list('abbbaca'),'data1':range(7)})
df2=DataFrame({'key':list('abc'),'data2':range(3)})
df2


# In[3]:


pd.merge(df1,df2)


# In[4]:


pd.merge(df1,df2,on = 'key')


# In[5]:


df3 = DataFrame({'lkey': ['b', 'b', 'a', 'c', 'a', 'a', 'b'],
                 'data1': range(7)})
df4 = DataFrame({'rkey': ['a', 'b', 'd'],
                 'data2': range(3)})
pd.merge(df3,df4,left_on = 'lkey',right_on = 'rkey')


# In[6]:


pd.merge(df1,df2,how = 'outer')


# In[7]:


df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                 'data1': range(6)})
df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
                 'data2': range(5)})
pd.merge(df1,df2,on = 'key',how = 'left')


# In[8]:


left = DataFrame({'key1': ['foo', 'foo', 'bar'],
                  'key2': ['one', 'two', 'one'],
                  'lval': [1, 2, 3]})
right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                   'key2': ['one', 'one', 'one', 'two'],
                   'rval': [4, 5, 6, 7]})
pd.merge(left,right,on = ['key1','key2'],how='inner')


# In[9]:


pd.merge(left,right,on = 'key1',suffixes=['_left','_right'])


# ## 索引上的合并

# In[10]:


left1 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'],
'value': range(6)})
right1 = DataFrame({'key':[3.5,7]},index = ['a','b'])
right1


# In[11]:


pd.merge(left1,right1,left_on = 'key',right_index = True)


# In[12]:


import numpy as np
lefth = DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'], 'key2': [2000, 2001, 2002, 2001, 2002], 'data': np.arange(5.)})
righth = DataFrame(np.arange(12).reshape((6, 2)), index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'], [2001, 2000, 2000, 2000, 2001, 2002]],columns=['event1', 'event2'])
righth


# In[13]:


pd.merge(lefth,righth,left_on=['key1','key2'],right_index = True,how = 'outer')


# In[14]:


left2 = DataFrame([[1., 2.], [3., 4.], [5., 6.]], index=['a', 'c', 'e'],columns=['Ohio', 'Nevada'])
right2 = DataFrame([[7., 8.], [9., 10.], [11., 12.], [13, 14]],index = ['b','c','d','e'],columns = ['Missouri','Alabama'])
right2


# In[15]:


left2.join(left1,how = 'outer')


# In[16]:


another = DataFrame([[7., 8.], [9., 10.], [11., 12.], [16., 17.]],
index=['a', 'c', 'e', 'f'], columns=['New York', 'Oregon'])
left2.join([right2,another])


# ## 轴向连接

# In[17]:


arr = np.arange(12).reshape((3,4))
arr


# In[18]:


np.concatenate([arr,arr],axis = 1)


# In[19]:


s1 = Series([0,1],index=['a','b'])
s2 = Series([2,3,4],index = ['c','d','e'])
s3 = Series([5,6],index = ['f','g'])
pd.concat([s1,s2,s3])


# In[20]:


pd.concat([s1,s2,s3],axis = 1)


# In[21]:


s4 = pd.concat([s1*5,s3])


# In[22]:


pd.concat([s1,s4],axis = 1,join_axes=[['a','c','b','e']])


# In[23]:


result = pd.concat([s1,s2,s3],keys = ['one','two','three'])
result.unstack()


# In[24]:


result = pd.concat([s1,s2,s3],keys = ['one','two','three'],axis = 1)
result


# In[25]:


df1 = DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'],columns=['one', 'two'])
df2 = DataFrame(5 + np.arange(4).reshape(2, 2), index=['a', 'c'],columns=['three', 'four'])
pd.concat({'level1':df1,'level2':df2},axis = 1)


# In[26]:


pd.concat({'level1':df1,'level2':df2},axis = 1,names = ['upper','lower'])


# In[27]:


df1 = DataFrame(np.random.randn(3,4),columns = ['a','b','c','d'])
df2 = DataFrame(np.random.randn(2,3),columns = ['b','d','a'])
pd.concat([df1,df2],ignore_index = True)


# ## 合并重叠数据

# In[28]:


a = Series([np.nan,2.5,np.nan,3.5,4.5,np.nan]
          ,index = list('fedcba'))
b = Series(np.arange(len(a), dtype=np.float64),
        index=['f', 'e', 'd', 'c','b','a'])
b[-1]


# In[29]:


result = np.where(a.isnull(),b,a) 
#result = np.where(pd.isnull(a),b,a) 
result


# ## 重塑和轴向旋转

# In[30]:


data = DataFrame(np.arange(6).reshape((2, 3)),
                 index=pd.Index(['Ohio', 'Colorado'], name='state'),
                 columns=pd.Index(['one', 'two', 'three'], name='number'))
data


# In[31]:


# data.unstack()
result = data.stack()
result


# In[32]:


result.unstack()


# In[33]:


result.unstack('state')


# ## 将长格式转化为宽格式

# In[34]:


f = open('F:/电影/数据分析/pydata-book-master/ch07/macrodata.csv')
df = pd.read_csv(f)
periods = pd.PeriodIndex(year = df.year,quarter = df.quarter,name = 'date') # 得到周期时间处理
data = DataFrame(df.to_records(),columns = pd.Index(['realgdp','unemp','infl']),
                index = periods.to_timestamp('D','end'))
data.head()


# In[35]:


ldata = data.stack().reset_index().rename(columns = {0:'value','level_1':'item'})
ldata.head()


# In[36]:


pivoted = ldata.pivot('date','item','value')
pivoted.head()


# In[37]:


ldata['value2'] = np.random.randn(len(ldata))
ldata[:10]


# ## 数据转换

# ### 移除重复数据

# In[38]:


data = DataFrame({'k1': ['one'] * 3 + ['two'] * 4,
                  'k2': [1, 1, 2, 3, 3, 4, 4]})
data


# In[39]:


data.duplicated()


# In[40]:


data.drop_duplicates()


# In[41]:


data.drop_duplicates('k1')


# ### 利用函数或映射进行数据转换

# In[42]:


data = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami',
                  'corned beef', 'Bacon', 'pastrami', 'honey ham',
                  'nova lox'],
                  'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})
data


# In[43]:


meat_to_animal ={
    'bacon':'pig',
    'pulled pork': 'pig',
    'pastrami': 'cow',
    'corned beef': 'cow',
    'honey ham': 'pig',
    'nova lox': 'salmon'
}


# In[44]:


data['animal'] = data['food'].map(str.lower).map(meat_to_animal)
data


# ### 替换值

# In[45]:


data = Series([1., -999., 2., -999., -1000., 3.])
data


# In[46]:


data.replace(-999,np.nan)


# In[47]:


data.replace([-999,-1000],np.nan)


# In[48]:


data.replace({-999:np.nan,-1000:np.nan})


# ### 重命名索引

# In[49]:


data = DataFrame(np.arange(12).reshape((3, 4)),
                 index=['Ohio', 'Colorado', 'New York'],
                 columns=['one', 'two', 'three', 'four'])
data.index.map(str.upper)


# In[50]:


data.rename(index = str.title,columns = str.upper)


# ### 离散化和面元划分

# In[51]:


age =  [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18,25,34,46,70]
cats = pd.cut(age,bins)
cats


# In[52]:


cats.codes


# In[53]:


pd.value_counts(cats)


# In[54]:


pd.cut(age, [18, 26, 36, 61, 100], right=False)


# In[55]:


group_names = ['Youth', 'YoungAdult', 'MiddleAged', 'Senior']
pd.cut(age,bins,labels=group_names)


# In[56]:


data = np.arange(20)
pd.cut(data,4,precision = 2)


# In[57]:


data = np.random.randn(1000)
cats = pd.qcut(data,4)
cats


# In[58]:


pd.value_counts(cats)


# In[59]:


pd.qcut(data,[0,0.1,0.5,0.9,1]) ## 自定义分位数


# ### 检测和过滤异常值

# In[60]:


np.random.seed(12345)
data = DataFrame(np.random.randn(1000,4))
data.describe()


# In[61]:


col = data[3]
col[np.abs(col) > 3]


# In[62]:


data[(np.abs(data) > 3).any(1)]


# In[63]:


data = np.sign(data)*3
data.describe()


# ### 排列随机取样

# In[64]:


df = DataFrame(np.arange(5*4).reshape((5,4)))
sampler = np.random.permutation(5)
sampler


# In[65]:


df.take(sampler)


# ### 计算哑变量/指标

# In[66]:


df = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                'data1': range(6)})
df


# In[67]:


pd.get_dummies(df.key)


# In[68]:


mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('F:/电影/数据分析/pydata-book-master/ch02/movielens/movies.dat',sep = '::',header = None,names = mnames)
movies[:10]


# In[69]:


gener_iter = (set(x.split('|')) for x in movies.genres)
generes = sorted(set.union(*gener_iter))
dummies = DataFrame(np.zeros((len(movies),len(generes))),columns = generes)
for i,gen in enumerate(movies.genres):
    dummies.loc[i,gen.split('|')] = 1
movies_windic = movies.join(dummies.add_prefix('Genre_'))


# In[70]:


movies_windic.iloc[0]


# In[71]:


movies_windic[:5]


# ## 字符串操作

# In[72]:


val = 'a,v,  guido '
result = val.split(',')
print(result)


# In[73]:


pieces = [x.strip() for x in result]
print(pieces)


# In[74]:


first,second,third = pieces
first+'::'+second+'::'+third


# In[75]:


'::'.join(pieces)


# In[76]:


'guido' in pieces


# In[77]:


val.find(',') # index方法寻找一个字符串，如果未果，返回错误，find方法返回-1


# In[78]:


val.count(',')


# In[79]:


val.replace(',','')


# ### 正则表达式

# In[80]:


import re
text = "foo   bar\t baz  \tqux"
re.split('\s+',text)


# In[81]:


text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com
"""
pattern = re.compile('(.*?)@(.*?)\.(\w{2,4})',re.IGNORECASE)
result = re.findall(pattern,text)
print(result)


# In[82]:


pattern = re.compile('(?P<username>.*?)@(?P<domain>.*?)\.(?P<suffix>w{2,4})',re.IGNORECASE|re.VERBOSE)
#pattern = re.compile('(?P<username>[A-Z0-9._%+-]+)@(?P<domain>[A-Z0-9.-]+) \.(?P<suffix>[A-Z]{2,4})',re.IGNORECASE|re.VERBOSE)
result = re.match(pattern,'wesm@bright.net')
print(result)


# ## 矢量化字符串函数

# In[83]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame
data = {'Dave': 'dave@google.com', 'Steve': 'steve@gmail.com', 'Rob': 'rob@gmail.com', 'Wes': np.nan}
data = Series(data)
data.isnull()


# In[84]:


data.str.contains('gmail')


# In[85]:


pattern


# In[86]:


data.str.findall(pattern)


# ### 练习

# In[87]:


import json 
json_file = open('F:/电影/数据分析/pydata-book-master/ch07/foods-2011-10-03.json')
db = json.load(json_file)
db[0].keys()


# In[88]:


nutrients = DataFrame(db[0]['nutrients'])
len(nutrients)


# In[89]:


nutrients[:4]


# In[90]:


info_keys = ['description','group','id','manufacturer']
info = DataFrame(db,columns = info_keys)
info[:6]


# In[91]:


pd.value_counts(info.group)[:10]


# In[92]:


nutrients = []

for rec in db:
    fnuts = DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)

nutrients = pd.concat(nutrients, ignore_index=True)
nutrients[:10]


# In[93]:


nutrients = nutrients.drop_duplicates()


# In[94]:


col_mapping = {'description':'food',
              'group':'fgroup'}
info = info.rename(columns=col_mapping,copy = False)


# In[95]:


col_mapping = {'description' : 'nutrient',
               'group' : 'nutgroup'}
nutrients = nutrients.rename(columns=col_mapping, copy=False)


# In[96]:


ndata = pd.merge(info,nutrients,on = 'id' , how = 'outer')
ndata[:4]


# In[97]:


result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)
result['Zinc, Zn'].sort_values().plot(kind = 'barh')


# In[98]:


result = ndata.groupby(['nutrient', 'fgroup'])

get_max = lambda x : x.xs(x.value.idxmax())
get_min = lambda x : x.xs(x.value.idxmin())
max_food = result.apply(get_max)[['value','food']]
max_food[:4]


# In[99]:


food_data = ndata.groupby(['food'])['value']
type(food_data)

