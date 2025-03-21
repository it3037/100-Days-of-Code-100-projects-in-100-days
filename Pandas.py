#!/usr/bin/env python
# coding: utf-8

# # Pandas
# 
# Pandas is the most popular python **Data Analysis & Data Structure** tool. 
# 
# ![image.png](attachment:image.png)
# 
# 
# - **Python with Pandas is used in a wide range of fields including academic and commercial domains including finance, economics, Statistics, analytics, etc**
# 
# ### Key Features of Pandas
# - Fast and efficient DataFrame object with default and customized indexing.
# - Tools for loading data into in-memory data objects from different file formats.
# - Data alignment and integrated handling of missing data.
# - Reshaping and pivoting of date sets.
# - Label-based slicing, indexing and subsetting of large data sets.
# - Columns from a data structure can be deleted or inserted.
# - Group by data for aggregation and transformations.
# - High performance merging and joining of data.
# - Time Series functionality.
# 
# 
# ## Data Analysis
# - Raw data - information- Prepare- Feature selection- Model Data 
# - import data(Data Acquistion) - Data prepartion(Cleaning data, Data Engineer) - EDA - Model Data 

# ### Installation
# Standard Python distribution doesn't come bundled with Pandas module. A lightweight alternative is to install pandas using popular Python package installer, pip.
# - pip install pandas
# - If you install Anaconda Python package, Pandas will be installed by default with the following −
#   Anaconda (from https://www.continuum.io) is a free Python distribution for SciPy stack.
#   
# #### Pandas deals with the following three data structures −
# 
# - Series: Series is a one-dimensional array like structure with homogeneous data.
# - DataFrame:DataFrame is a two-dimensional array with heterogeneous data
# - Panel: Panel is a three-dimensional data structure with heterogeneous data. It is hard to represent the panel in graphical representation. But a panel can be illustrated as a container of DataFrame.
# These data structures are built on top of Numpy array.

# - Data Acquition: , - Data Prepare: - Data Cleanings, Data manipulation, - Data Enginerring 
#         Raw Data - Information - Insights- Actions 
# - pip install pandas 
# import pandas 

# - **Lets work with Series** 
# 1. Series (1D) , rows , 
# 2. Data Frames (2D) , rows and columns 
# 3. Panel (3D)

# In[1]:


import pandas as pd
import numpy as np


# In[6]:


a=(1,2,3,4,5,6)

b = ["Anand","Amgad","Karim","Sherine","Random","TKA"] 
#dtype=int
x = pd.Series(data = b)
x


# In[7]:


y = pd.Series(data = a,index = b,dtype=int)
y


# In[5]:


y['four']


# In[8]:


#Series from list
a = [1,2,3,4]
b = ("I","II","III","IV")
s = pd.Series(data = a, index= b ,dtype=int)
s


# In[9]:


s.index


# In[10]:


s['III']


# In[11]:


data = pd.Series([1,2,3,4.6])
data


# In[12]:


data[::-1]


# In[13]:


data[-2] ## NEGATIVE INDEX DOESNT WORK HERE


# In[14]:


data[2:4]


# In[15]:


data[2]


# In[16]:


data.index


# In[17]:


data.values


# - The Pandas Series is much more general as well as flexible as compared to 1‐D NumPy array that it emulates Series as generalized NumPy array
# 
# - The Series object is basically interchangeable with a 1‐D NumPy array
# - The significant difference is the presence of the index: whereas the Numpy Array has an implicitly defined integer index used in order to obtain the values, the Pandas Series has a clear‐cut defined index associated with the values
# - The Series object additional capabilities are provided by this clear index description.The index needs not to be an integer but can made up of values of any wanted type. For instance, we can use strings as an index:

# # Series as specialized Dictionary

# - A dictionary is a structure which maps arbitrary keys to a collection of arbitrary values, as well as a Series is a structure which maps typed keys to a set of typed  values
# - This typing is significant: just as the type‐specific compiled code behind a NumPy array makes it more well‐organized than a Python list for certain operations, the type information of a Pandas Series makes it much more efficient as compare to Python dictionaries for certain operations
# - By creating a Series object directly from a Python dictionary the Series‐as‐dictionary analogy can be made even more explicit:

# ##  Creating series from dictionary

# In[18]:


emp={"A":8,"B":9,"C":10}
details=pd.Series(emp)
emp,details 


# - **Note: Values are used by default  as series  elements & Keys as index**
# Dictionary is a mapping data type , We cannot manupulate index in as we do in case of List & Tuples.

# In[19]:


#changing order of index
age = {'ram' : 28,'bob' : 19, 'cam' : 22}

s = pd.Series(data= age,index=['bob','ram','cam',"hello"],dtype=int)
s 


# - **note : Missing value is filled by NAN & index taken by keys**

# In[21]:


population_dict ={'California':38332521,
                 'Texas':26448193,
                 'New York':19651127,
                 'Florida':19562860,
                 'Florida':19662860,
                 'Illinois':12882135,
                 'Illinois':12892135}
population_series = pd.Series(population_dict)
population_dict,population_series


# In[22]:


population_series['Texas':'Illinois']


# In[23]:


population_dict.keys()


# In[24]:


population_dict.values()


# In[25]:


population_series['New York']


# In[26]:


population_series_rev = pd.Series(data = population_dict.keys() , 
                              index = population_dict.values(),
                              dtype = str)
population_series_rev                 


# In[27]:


pd.Series([2,3,4])


# In[28]:


pd.Series(5,index=[100,200,300])


# In[53]:


pd.Series({2:'a',1:'b',3:'c'})


# In[29]:


pd.Series({2:'a',1:'b',3:'c'},index=[3,2])


# - **Lets Work with Data Frame**

# The Pandas DataFrame Object
# - In Pandas, the next primary structure is the DataFrame
# - The DataFrame can be examined either as a Python dictionary specialisation or a generalization of a NumPy array
# 
# - DataFrame as a generalized NumPy array
# - Suppose a Series is an analogue of a 1‐D array with flexible indices. In that case, a DataFrame is an analogue of a 2‐D array with both flexible column names and flexible row indices
# - For showing this, first, make a new Series listing the area of each of the five states:
# 
# 
# - Therefore we can think DataFrame as a generalization of a 2‐D NumPy array, where
# both the rows and columns have a generalized index to access the data
# 
# - DataFrame as specialized dictionary
# - Likewise, we can consider a DataFrame as a specialization of a dictionary as well. 
# - Where a DataFrame maps a column name to a Series of column data, a dictionary maps a key to a value

# In[30]:


area_dict = {'California':423967,
            'Texas':695662,
            'New York':141297,
            'Florida':170312,
            'Illinois':149995}
area_series = pd.Series(area_dict) 
area_series


# In[31]:


states = pd.DataFrame({'population':population_series,
                      'area':area_series})
states


# In[32]:


states.index


# In[33]:


states.values


# In[34]:


states.columns


# In[35]:


states['area'] ## in order to read column from a dataset use square bracket along with column name in single quotes


# - Data[0] will return the first row in a 2‐D NumPy array. Data['col0'] will return the first
# column for a DataFrame

# In[68]:


import pandas as pd
# Data frame from 1D List
l=["ashi","rom","sid"]
df=pd.DataFrame(l)
df 


# In[71]:


#2D lIST
data = [['Nokia',10000],['Asus',12000],['Samsung',13000],['Apple',33000]]
d = pd.DataFrame(data,columns=['Mobile','Price'],index=[1,2,3,4])
d


# # Constructing DataFrame objects

# - Various ways can be used in order to construct Pandas DataFram. The following are
# several examples:
# - From a single Series object: 
# - A DataFrame is a collection of Series objects.
# - Moreover, from a single Series a single‐column DataFrame can be constructed

# In[76]:


pd.DataFrame(population_series, columns = ['population'])


# - From a list of dicts: Any list of dictionaries can be made into a DataFrame

# In[77]:


data = [{'a':i,'b':2 * i} for i in range(5)]
pd.DataFrame(data)


# - Even if a few keys are missing in the dictionary, they will be filled by Pandas with
# NaN which means "not a number" values:

# In[82]:


pd.DataFrame([{'a':1,'b':2},{'b':'Ankita','c':4},{'c':'Ankur','d':2}])


# - From a dictionary of Series objects: A DataFrame can be constructed from a
# dictionary of Series objects as well:

# In[80]:


pd.DataFrame({'population':population_series,
                      'area':area_series})


# ## Create a data frame containing Details of six students in chem, phy, maths,Roll/name(Index)

# **note : If no index or column  is passed, then by default, index will be range(n), where n is the array length.**

# In[85]:


import pandas as pd

data = [["A",34,23,56],["B",92,88,76],["C",76," ",25],["D"," ",78,99]]

stu = pd.DataFrame(data,columns=["Name",'Maths','Phy','Chem'],index=[1,2,3,4],dtype=int)
stu 


# In[88]:


stu.replace(" ",0)


# ### Creating Data frame from Series

# In[94]:


import pandas as pd
# Selecting Columns
d = {'Chem' : pd.Series([30, 70, 35,25,26,67,77,67,89], index=["Raj",'Ram', 'Asa', 'Pi','Chi','Ru',"Sita","Ria","Gita"],dtype=int),
     'Math' : pd.Series([18, 26, 35, 40,55,89,79,100], index=["Raj",'Ram', 'Pi', 'Chi', 'Ru',"Sita","Ria","Gita"],dtype=int),
     'Phy' : pd.Series([31, 42, 83,34,80,78], index=["Asa",'Ram', 'Pi', 'Ru',"Sita","Gita"],dtype=int)}
exam = pd.DataFrame(d)
exam1=exam.copy() ## To make copy of your data
exam1


# ## Data Preparation
# - Removing null values/replace
# - Data description 
# - Adding new data fields:analysis 
# - Feature selection: ML , Predictions:decisions 

# In[95]:


# Adding columns

print ("Adding a new column using the existing columns in DataFrame:")
exam1['Total']=exam1['Chem']+exam1['Math']+exam1['Phy']
exam1 


# ## Dealing with missing data
# - Check for Missing Values
# 
# - To make detecting missing values easier (and across different array dtypes), Pandas provides the isnull() and notnull() functions, which are also methods on Series and DataFrame objects
# 
# 

# In[96]:


exam


# In[7]:


exam.isnull().sum()

exam.isnull()#to check null values 
# ### Calculations with Missing Data
# 
# - When summing data, NA will be treated as Zero If the data are all NA, then the result will be NA
# Cleaning / Filling Missing Data
# 
# - Pandas provides various methods for cleaning the missing values. The fillna function can “fill in” NA values with non-null data in a couple of ways, which we have illustrated in the following sections.
# 
# - Replace NaN with a Scalar Value The following program shows how you can replace "NaN" with "0".
# 

# In[98]:


exam=exam.fillna(0)
exam 


# In[99]:


exam["T"]=exam["Math"]+exam["Phy"]+exam["Chem"]
exam


# In[100]:


exam1

median(1,2,3)
mode(1,2,2,3,4,): Categorical data
median,mean(Con): median(Is not afftected by outliers)
1,1,2,3,4,5,67,89,8,9
1,1,2,3,4,5,8,9,67,89 
# In[104]:


a = exam1["Math"].median()

exam1["Math"]= exam1["Math"].fillna(a)
exam1 

b=exam1["Phy"].median()
exam1["Phy"]= exam1["Phy"].fillna(b)

exam1["Total_Cleaned"]=exam1["Math"]+exam1["Phy"]+exam1["Chem"]
exam1


# ## Dropping missing values using "dropna()"
# 
# - 10000 
# A- 999(NA): Drop A
# B-79(NA): Replace NA
# C- 0(NA)

# In[105]:


exam
exam=exam.dropna()
#exam_1
## dataframe copy
## exam2=exam.copy()
exam
##drop exam["Phy"]


# ### Replacing missing values by generic values by replace function

# In[107]:


df = pd.DataFrame({'one':[10,20,30,40,50,"ABC"], 'AGE':[-19,1,30,40,50,60]})
df


# In[108]:


df_cleaned = df.replace({"ABC":60,-19:19})
df_cleaned


# ### Stats: Data Description

# In[ ]:


import pandas as pd
details = {'Brand':pd.Series(['Nokia','Asus',"Nokia","Nokia",'Samsung',"ABC",'Micromax','Apple','MI','Zen',"Apple"]),
   'Price':pd.Series([10000,8000,12500,7000,40000,12000,12999,13999,59999]),
   'Rating(10)':pd.Series([7,6.5,8.5,9,8,9.5,7,9])
}

d = pd.DataFrame(details)
d


# In[ ]:


d.mean()


# In[ ]:



d["Price"].mean()


# In[ ]:


#d["Rating(10)"].describe()
d.describe() 


# ### The describe() function computes a summary of statistics pertaining to the DataFrame columns.

# In[ ]:


d.describe(include="all")
#d[d['Brand'] == 'Apple']
#d.head(2)#displays data labels from start
#d.tail(1)# dispalys data lables from bottom
# - Mode,Sum,medain,valuable(null), any details about categorical


# ## Lets Practice
# - Check null values for the given data
# - replace null values by mean/median/mode 
# - describe

# ### Renaming
# - The rename() method allows you to relabel an axis based on some mapping (a dict or Series) or an arbitrary function.

# In[ ]:


d


# In[ ]:



d=d.rename(columns={'Brand' : 'Type', 'Avg.Price' : 'Price'},
index = {0 : 'S0', 1 : 'S1', 2 : 'S2'})
d


# In[ ]:


x=d["Price"].mean()
d["Price"]=d["Price"].fillna(x)
d["Rating(10)"]=d["Rating(10)"].fillna(d["Rating(10)"].mean())
d


# In[ ]:


d.isnull().sum()


# ## Sorting 

# In[16]:


import pandas as pd
details = {'Brand':pd.Series(['Nokia','Asus',"Nokia","Nokia",'Samsung',"ABC",'Micromax','Apple','MI','Zen',"Apple"]),
   'Avg.Price':pd.Series([10000,8000,12500,7000,40000,12000,12999,13999,59999]),
   'Rating(10)':pd.Series([7,6.5,8.5,9,8,9.5,7,9])
}

d = pd.DataFrame(details)
d


# In[17]:



df2 = d.sort_values(by=['Rating(10)'],ascending= False)#Decending order
df2 


# ## get_dummies()
# 
# - Pass a list with length equal to the number of columns.
# - Returns the DataFrame with One-Hot Encoded values.

# CAR PRICE VS "MODEL", MILAGE 
# 
# 
# 
s = pd.Series(['Orange ', ' Pink', 'Blue'])
"Orange":0, "Pink":1,"Blue":2 

   1- a
   2-b
   3-a    
   1a 2b 3a 
  a 1  0   1
  b 0   1  0 
  
  A-0
  B-1
  C-2
  D-3
  A-1,OTHERS ARE 0
  B-1,OTHERS ARE 0
# In[18]:


import pandas as pd                                                                                                                                  
s = pd.Series(['Orange', 'Pink', 'Blue'])
s1=s.str.get_dummies()
s1 


# In[19]:


d


# In[20]:



pd.get_dummies(d.Brand)


# ## Indexing & Selecting Data
# - loc() : (label based indexing) takes two single/list/range operator separated by ','. The first one indicates the row and the second one indicates columns.
# - iloc() : integer based indexing. 
# ## loc()

# In[3]:


import pandas as pd
details = {'Brand':pd.Series(['Nokia','Asus',"Nokia","Nokia",'Samsung',"ABC",'Micromax','Apple','MI','Zen',"Apple"]),
   'Avg.Price':pd.Series([10000,8000,12500,7000,40000,12000,12999,13999,59999,7899]),
   'Rating(10)':pd.Series([7,6.5,8.5,9,8,9.5,7,9,10])
}

d = pd.DataFrame(details)
d


# In[12]:


d.loc[:,"Rating(10)"]
d.iloc[0]


# In[ ]:


d.loc[3,["Brand","Rating(10)"]]
d.loc[:,"Brand"]
## d.loc["Nokia"]


# In[ ]:


d.loc[0]# Fetching a member by index name


# In[ ]:


# Select range of rows for all columns
d.loc[2:5]


# ####  About iloc() : 

# In[ ]:



# select all rows for a specific column
d.iloc[:,2]


# In[ ]:


d.iloc[2:5]


# In[ ]:


# Integer slicing
d.iloc[2:5, 1:3]


# In[ ]:


# Slicing through list of values
d.iloc[[7],[0]]

### Using index parameter
df = pd.DataFrame(np.random.randn(8, 4), columns = ['A', 'B', 'C', 'D'])
print (df['A'])
print (df[['A','B']])
print( df[2:2])
## Attribute Access
d.Brand
# ### Correlation
# Correlation shows the linear relationship between any two array of values (series). There are multiple methods to compute the correlation like pearson(default), spearman and kendall.

# ![image.png](attachment:image.png)

# - price increases with years (+ive)
# - For ABC cmpny Sales is decreasing with Time (-ve)
# - In last 5 years toursim industry has a constant growth of 0.6 per. 
# - 30 marks : 12 hrs , 12 marks : 5 hrs 
# 
# ![image.png](attachment:image.png)
d.corr()
-1,1
-1: one increases other varaible will decrease 
1: one 
0: not related 
# In[13]:


import numpy as np                                                 # Implemennts milti-dimensional array and matrices
import pandas as pd                                                # For data manipulation and analysis
#import pandas_profiling
import matplotlib.pyplot as plt                                    # Plotting library for Python programming language and it's numerical mathematics extension NumPy
#import seaborn as sns                                              # Provides a high level interface for drawing attractive and informative statistical graphics
get_ipython().run_line_magic('matplotlib', 'inline')
#sns.set()
from subprocess import check_output
d.corr()
d.plot(x="Avg.Price",y="Rating(10)",style="*")
plt.show()


# In[14]:


d.corr()

### Percent_change
Series, DatFrames and Panel, all have the function pct_change(). This function compares every element with its prior element and computes the change percentage.
s = pd.Series([1,2,4,8,6,4])
print (s.pct_change())

# ### Creating joints in Pandas
# - **Full Outer Join**
# 
#  combines the results of both the left and the right outer joins. The joined  DataFrame will contain all records from both the DataFrames and fill in NaNs for missing matches on either side. You can perform a full outer join by specifying the how argument as outer in the merge() function:
#  
# - **Inner Join**
#  
#  combines the common results of both
#  - There should be relevance(Common field,analysis requirement)

# In[25]:


import pandas as pd
d = {
        'id': ['1', '2', '3', '4', '5','6'],
        'Color': ['RED', 'GREEN', 'YELLOW', 'BLUE', 'PINK','BLACK'],
        'Fruit': ['APPLE', 'BANANA', 'MANGO', 'BERRY', 'MELON','GRAPES']}
d1=pd.DataFrame(d)
d1


# In[26]:


z = {    'id': ['1', '2', '3', '4', '5','7'],
        'rating': ['A', 'B', 'C', 'A', 'B','A'],
        'COST': [200, 230, 400, 400, 100,450],
        'Fruit': ['APPLE', 'BANANA', 'MANGO', 'BERRY', 'MELON','KIWI'],
        'BUY': ['Y', 'N', 'Y', 'N', 'Y','N']}
d2=pd.DataFrame(z)
d2


# In[36]:


join = pd.merge(d1, d2, on="Fruit", how='right') ## similary we can  use 'outer/right/left join'

join 


# In[38]:


#Selecting few fields from frames
join = pd.merge(d1[["Color","Fruit"]], d2, on="Fruit", how='outer') ## similary we can  use 'inner/right/left join'

join


# In[33]:


#Selecting both common fields from frames
join = pd.merge(d1, d2, on=["id","Fruit"], how='outer') ## similary we can  use 'inner/right/left join'

join


# ### How simple merge function differs from join?
# Simple merge works as an inner joint

# In[39]:


df_merge = pd.merge(d1,d2, on='Fruit')

df_merge 


# ## Concadination

# In[43]:


a = {
        "id": [12,14],
        'Color': ['Brown', 'yellow', ],
        'Fruit': ['Pear', 'muskmellon']}
d3=pd.DataFrame(a)
d3


# In[47]:


df_col = pd.concat([d1,d3], axis=0)

df_col.reset_index()
#df_col


# ## Frequency table : Crosstab

# In[48]:


d2


# In[52]:


my_tab = pd.crosstab(index=d2["rating"],  # Make a crosstab
                              columns="c1")      # Name the count column

my_tab 


# In[ ]:


c = pd.value_counts(d2.rating).to_frame().reset_index()
c


# ### Split Data into Groups

# In[53]:


import pandas as pd
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings',
   'Kings', 'Kings', 'Kings', 'Riders', 'Royals',"MI", 'Royals', 'Riders',"MI","MI"],
   'Rank': [1, 2, 2, 3, 3,4 ,1 ,1,2 ,3, 4,1,2,1,1],
   'Year': [2014,2015,2014,2015,2014,2015,2016,2017,2016,2016,2014,2015,2017,2018,2019],
   'Points':[876,789,863,673,741,812,756,788,694,701,607,804,690,890,900]}
ipl = pd.DataFrame(ipl_data)
ipl.shape
#print (ipl.groupby('Team'))


# In[54]:


ipl


# In[55]:


c=ipl.groupby('Team').groups
c


# In[56]:


a=ipl.groupby(['Team','Year']).groups
a


# In[71]:


team_data = ipl.groupby(["Team",'Year'])# Similary try for Year ,Rank & Points
team_data.get_group(("Devils",2014))


# In[64]:


ipl[ipl["Team"]=="Devils"]


# In[61]:


my_tab = pd.crosstab(index=ipl["Team"],  # Make a crosstab
                              columns="count")      # Name the count column

my_tab


# In[ ]:


## Agrregration on groups
import numpy as np
grouped = ipl.groupby('Year')
grouped['Points'].agg(np.mean)


# In[ ]:


# Attribute Access in Python Pandas
import numpy as np
grouped = ipl.groupby('Team')
print (grouped.agg(np.size))


# In[ ]:


grouped = ipl.groupby('Team')
print (grouped['Points'].agg([np.sum, np.mean, np.std]))


# ### Filtration
# - Filtration filters the data on a defined criteria and returns the subset of data. The filter() function is used to filter the data.
# 
# ### Lambda function
# - The lambda keyword is used to create anonymous functions
# - This function can have any number of arguments but only one expression, which is evaluated and returned.
# - One is free to use lambda functions wherever function objects are required.
# - You need to keep in your knowledge that lambda functions are syntactically restricted to a single expression.
# 

# In[ ]:


x=lambda a,b,c : (a*b/c)
print(x(5,3,2)) 


# In[ ]:


a=[1,3,4,6,23,44,56,78,90,54,60]
list_e = list(filter(lambda x:(x %2==0),a))
print(list_e)


# In[ ]:


##Lambda functions can take any number of arguments
x = lambda a, b : a * b
x(5, 6)
x(8,7)


# In[ ]:


# Python code to illustrate 
# filter() with lambda() 
final_list = list(filter(lambda x: (x < 800) , ipl["Points"])) 
print(final_list)


# In[ ]:


ipl[ipl["Rank"].apply(lambda s: s == 1)] ## by directly Acessing Colums 


# In[ ]:


import pandas as pd
top=lambda x:x ==1
ipl[ipl["Rank"].apply(top)] ## by column name


# In[ ]:


cap = lambda x: x.upper()
ipl['Team'].apply(cap)


# ### datetime.now() gives you the current date and time.

# In[ ]:


print (pd.datetime.now())


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from time import sleep\nsleep(5)\n3+4')


# ## Input Output

# ## Reading files

# ### Following files can be read in pandas:
# 
# - df.to_csv(...)  # csv file
# - df.to_hdf(...)  # HDF5 file
# - df.to_pickle(...)  # serialized object
# - df.to_sql(...)  # to SQL database
# - df.to_excel(...)  # to Excel sheet
# - df.to_json(...)  # to JSON string
# - df.to_html(...)  # render as HTML table
# - df.to_feather(...)  # binary feather-format
# - df.to_latex(...)  # tabular environment table
# - df.to_stata(...)  # Stata binary data files
# - df.to_msgpack(...)	# msgpack (serialize) object
# - df.to_gbq(...)  # to a Google BigQuery table.
# - df.to_string(...)  # console-friendly tabular output.
# - df.to_clipboard(...) # clipboard that can be pasted into Excel
import os
print(os.getcwd())
# In[1]:


get_ipython().run_cell_magic('time', '', 'import pandas as pd\n#d=pd.read_excel("movies.xlsx")\n\nviz_data=pd.ExcelFile(r"C:\\Users\\suyashi144893\\Documents\\data Sets\\Exam_data.xlsx").parse("StudentsPerformance")\nviz_data')


# In[ ]:


viz_data.head(3)


# In[ ]:


import pandas as pd
tp_d=pd.read_excel("movies.xlsx")
tp_d.head(3)


# In[ ]:


movie_data.head()
movie_data.shape


# In[ ]:


movie_data.drop_duplicates()


# ## This will fectch data from bottom

# In[ ]:


##Movie Trend 2006 - 2016
1. Describe data
2. Drop Columns
3. Check Null values 
4. Replace null values 


# In[ ]:


import pymysql
query = open('my_data.sql', 'r')
con= "F:\python"
DF = pd.read_sql_query(query.read(),con)


# In[ ]:


Emp.describe()


# - SQL DATA FROM SERVER

import pandas as pd
from sqlalchemy import create_engine

def getData():
  # Parameters
  ServerName = "my_server"
  Database = "my_db"
  UserPwd = "user:pwd"
  Driver = "driver=SQL Server Native Client 11.0"

  # Create the connection
  engine = create_engine('mssql+pyodbc://' + UserPwd + '@' + ServerName + '/' + Database + "?" + Driver)

  sql = "select * from mytable"
  df = pd.read_sql(sql, engine)
  return df

df2 = getData()
print(df2)
# ## creating a csv file

# In[ ]:


empid=[100,200,300,400]
emprole=["lead","Trainer","Consultant","Sales"]
details=list(zip(empid,emprole))
details


# In[ ]:


import pandas as pd
df=pd.DataFrame(data=details,index=["ONE","TWO","THREE","FOUR"])
df


# In[ ]:


df.to_csv("e.csv",header=False,index=False)


# ## Pandas SQL Operations
# We can use the pandas read_sql_query function to read the results of a SQL query directly into a pandas DataFrame. The below code will execute the same query that we just did, but it will return a DataFrame. It has several advantages over the query we did above:
# 
# - It doesn’t require us to create a Cursor object or call fetchall at the end.
# - It automatically reads in the names of the headers from the table.
# - It creates a DataFrame, so we can quickly explore the data.

# In[ ]:


conda  install -c sqlite3


# In[ ]:


# Program to Connect to the exixting database
##If the database does not exist, then it will be created and finally a database object will be returned
import sqlite3

conn = sqlite3.connect('D1.db')

print ("My first Connection")


# In[ ]:


# Program to create a table in the previously created database
conn = sqlite3.connect('D1.db')
print ("Opened database successfully")

conn.execute('''CREATE TABLE COMPANY2
         (ID INT PRIMARY KEY     NOT NULL,
         NAME           TEXT    NOT NULL,
         AGE            INT     NOT NULL,
         ADDRESS        CHAR(50),
         SALARY         REAL);''')
print ("Table created......")

conn.close()


# In[ ]:


## to create records in the COMPANY table created in the above example.
conn = sqlite3.connect('D1.db')
print ("Opened database successfully");

conn.execute("INSERT INTO COMPANY2 (ID,NAME,AGE,ADDRESS,SALARY)       VALUES (7, 'Poy', 30, 'UP', 30000.00 )")

conn.execute("INSERT INTO COMPANY2(ID,NAME,AGE,ADDRESS,SALARY)       VALUES (8, 'Ram', 33, 'FARIDABAD', 18000.00 )")

conn.execute("INSERT INTO COMPANY2 (ID,NAME,AGE,ADDRESS,SALARY)       VALUES (9, 'edd', 42, 'NEW DELHI', 22000.00 )")

conn.commit()
print ("Records updated successfully");
conn.close()


# In[ ]:


import pandas as pd
import sqlite3
conn = sqlite3.connect("D1.db")
Q = pd.read_sql_query(
'''SELECT id, name, address, salary from COMPANY2''', conn)


# In[ ]:


d = pd.DataFrame(Q)#, columns=['ID','NAME','ADDRESS','SALARY'])


# In[ ]:


d

