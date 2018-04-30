
# coding: utf-8

# # 数据加载、存储与文件格式

# ##  文本格式的数据

# In[1]:


from pandas import DataFrame,Series
import pandas as pd


# In[2]:


file1 = open('F:/电影/数据分析/pydata-book-master/ch06/ex1.csv')
type(file1)


# In[3]:


df = pd.read_csv(file1)
df


# In[4]:


file2 = open('F:/电影/数据分析/pydata-book-master/ch06/ex2.csv')
df2 = pd.read_csv(file2,names=['a','b','c','d','message'],index_col='message')
names = ['a','b','c','d','message']
df2


# In[5]:


file3 = open('F:/电影/数据分析/pydata-book-master/ch06/csv_mindex.csv')
df3 = pd.read_csv(file3,index_col=['key1','key2'])
df3


# In[6]:


file4 =open('F:/电影/数据分析/pydata-book-master/ch06/ex3.txt')
result = pd.read_table(file4,sep='\s+')
result


# In[7]:


file4 = open('F:/电影/数据分析/pydata-book-master/ch06/ex4.csv')
df4 = pd.read_csv(file4,skiprows=[0,2,3])
df4


# In[8]:


file5 = open('F:/电影/数据分析/pydata-book-master/ch06/ex5.csv')
df5 = pd.read_csv(file5)
df5


# In[9]:


file6 = open('F:/电影/数据分析/pydata-book-master/ch06/ex6.csv')
df6 =pd.read_csv(file6,nrows = 10)
df6


# ## 逐块读取文本数据

# In[10]:


file6 = open('F:/电影/数据分析/pydata-book-master/ch06/ex6.csv')
chunker = pd.read_csv(file6,chunksize=1000)
chunker


# In[11]:


file6 = open('F:/电影/数据分析/pydata-book-master/ch06/ex6.csv')
chunker = pd.read_csv(file6,chunksize=1000)
tot = Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)

tot = tot.sort_values(ascending=False)
tot[:10]


# ## 导出或者打印数据

# In[12]:


import sys
file6 = open('F:/电影/数据分析/pydata-book-master/ch06/ex6.csv')
data = pd.read_csv(file6)
data.to_csv('F:/电影/数据分析/pydata-book-master/ch06/ooo.csv')
#data.to_csv(sys.stdout, na_rep='NULL',index=False,column=False)
# data.to_csv(sys.stdout, sep='|')


# In[13]:


kk = Series.from_array('F:/电影/数据分析/pydata-book-master/ch06/tseries.csv')
kk


# ## 手工处理分隔符

# In[14]:


import csv
f = open('F:/电影/数据分析/pydata-book-master/ch06/ex7.csv')
reader = csv.reader(f)
reader


# In[15]:


for line in reader:
    print(line)


# In[16]:


lines = list(csv.reader(open('F:/电影/数据分析/pydata-book-master/ch06/ex7.csv')))
header, values = lines[0], lines[1:]
data_dict = {h:v for h,v in zip(header,zip(*values))}
data_dict


# In[17]:


from pandas import DataFrame,Series
import pandas as pd
df = DataFrame(data_dict,columns = ['a','b','c'])
df


# In[18]:


class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar = '"'
    quoting = 0
with open('F:/电影/数据分析/pydata-book-master/ch06/mydata.csv','w') as f:
    writer = csv.writer(f,dialect = my_dialect)
    writer.writerow(('one', 'two', 'three'))
    writer.writerow(('1', '2', '3'))
    writer.writerow(('4', '5', '6'))
    writer.writerow(('7', '8', '9'))


# ## Json数据

# In[19]:


obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
                 {"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""
import json 
result = json.loads(obj)
result


# In[20]:


asjson = json.dumps(result)
asjson


# In[21]:


siblings = DataFrame(result['siblings'],columns=['name','age','pet'])
siblings


# ## XML和HTML,web数据收集

# In[ ]:


from lxml.html import parse
from urllib.request import urlopen

parsed = parse(urlopen('http://www.sse.com.cn/market/stockdata/activity/'))

doc = parsed.get_root()


# ## 二进制数据格式

# In[23]:


pd.read_pickle('F:/电影/数据分析/pydata-book-master/ch06/frame_pickle')


# ## 读取Excel数据

# In[26]:


xls_file = pd.ExcelFile('F:/电影/数据分析/炼数成金-数据分析/第9周/Amtrak.xls')
table = xls_file.parse('StatProForecast1')
table


# ## 使用数据库

# In[28]:


import sqlite3
query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
c REAL,               d INTEGER
);"""
con = sqlite3.connect(':memory:')
con.execute(query)
con.commit()


# In[31]:


data = [('Atlanta', 'Georgia', 1.25, 6),('Tallahassee', 'Florida', 2.6, 3),('Sacramento', 'California', 1.7, 5)]
stmt = "Insert  into test Values(?,?,?,?)"
con.executemany(stmt,data)
con.commit()


# In[33]:


cursor = con.execute('select * from test')
rows = cursor.fetchall()
rows


# In[38]:


import pandas.io.sql as sql
sql.read_sql_query('select * from test', con)

