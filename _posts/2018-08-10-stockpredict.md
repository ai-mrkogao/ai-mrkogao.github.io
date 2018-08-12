---
title: "Stock Predict Experiment 1"
date: 2018-08-10
classes: wide
use_math: true
tags: economic index python stock utils kospi keras tensorflow pandas numpy
category: stock
---


# Decide Economic Index


# Python Tutorial

## Install xlrd
```bash
pip3 install xlrd
```

## Read Excel 

```python
xl = pd.ExcelFile("dummydata.xlsx")
xl.sheet_names
>>[u'Sheet1', u'Sheet2', u'Sheet3']

df = xl.parse("Sheet1")
df.head()
```

```python
parsed = pd.io.parsers.ExcelFile.parse(xl, "Sheet1")
parsed.columns
```

```python
import pandas
df = pandas.read_excel(open('your_xls_xlsx_filename','rb'), sheetname='Sheet 1')

# or using sheet index starting 0
df = pandas.read_excel(open('your_xls_xlsx_filename','rb'), sheetname=2)

```

```python
import pandas as pd
# open the file
xlsx = pd.ExcelFile(FileName.xlsx)
# get the first sheet as an object
sheet1 = xlsx.parse(0)
# get the first column as a list you can loop through
# where the is 0 in the code below change to the row or column number you want    
column = sheet1.icol(0).real
# get the first row as a list you can loop through
row = sheet1.irow(0).real
```

```python
import pandas as pd
# Read the excel sheet to pandas dataframe
DataFrame = pd.read_excel("FileName.xlsx", sheetname=0)
```
- read excel with specific row, column

```python
import pandas as pd

# define the file name and "sheet name"
fn = 'Book1.xlsx'
sn = 'Sheet1'

data = pd.read_excel(fn, sheetname=sn, index_col=0, skiprows=1, header=0, skip_footer=1)
```

## Transpose column
```python
>>> d1 = {'col1': [1, 2], 'col2': [3, 4]}
>>> df1 = pd.DataFrame(data=d1)
>>> df1
   col1  col2
0     1     3
1     2     4

>>> df1_transposed = df1.T # or df1.transpose()
>>> df1_transposed
      0  1
col1  1  2
col2  3  4
```

## Rename Column
```python
>>> df = pd.DataFrame({'$a':[1,2], '$b': [10,20]})
>>> df.columns = ['a', 'b']
>>> df
   a   b
0  1  10
1  2  20
```

```python
df.rename(columns={'pop':'population',
                          'lifeExp':'life_exp',
                          'gdpPercap':'gdp_per_cap'}, 
                 inplace=True)
```

## Create DateTimeIndex in Pandas
```python
import datetime as dt
import pandas as pd

df = pd.DataFrame({'year': [2015, 2016], 
                   'month': [12, 1], 
                   'day': [31, 1], 
                   'hour': [23, 1]})

# returns datetime objects
df['Timestamp'] = df.apply(lambda row: dt.datetime(row.year, row.month, row.day, row.hour), 
                           axis=1)

# converts to pandas timestamps if desired
df['Timestamp'] = pd.to_datetime(df.Timestamp)

>>> df
   day  hour  month  year           Timestamp
0   31    23     12  2015 2015-12-31 23:00:00
1    1     1      1  2016 2016-01-01 01:00:00

# Create a DatetimeIndex and assign it to the dataframe.
df.index = pd.DatetimeIndex(df.Timestamp)

>>> df
                     day  hour  month  year           Timestamp
2015-12-31 23:00:00   31    23     12  2015 2015-12-31 23:00:00
2016-01-01 01:00:00    1     1      1  2016 2016-01-01 01:00:00
```

## How to extract specific content in a pandas dataframe with a regex?
```python
#convert column to string
df['movie_title'] = df['movie_title'].astype(str)

#but it remove numbers in names of movies too
df['titles'] = df['movie_title'].str.extract('([a-zA-Z ]+)', expand=False).str.strip()
df['titles1'] = df['movie_title'].str.split('(', 1).str[0].str.strip()
df['titles2'] = df['movie_title'].str.replace(r'\([^)]*\)', '').str.strip()
print df
          movie_title      titles      titles1      titles2
0  Toy Story 2 (1995)   Toy Story  Toy Story 2  Toy Story 2
1    GoldenEye (1995)   GoldenEye    GoldenEye    GoldenEye
2   Four Rooms (1995)  Four Rooms   Four Rooms   Four Rooms
3   Get Shorty (1995)  Get Shorty   Get Shorty   Get Shorty
4      Copycat (1995)     Copycat      Copycat      Copycat
```

```python
value = re.sub(r"[^0-9]+", "", value)
df['pricing'] = re.sub(r"[^0-9]+", "", df['pricing'])
df['Pricing'].replace(to_replace='[^0-9]+', value='',inplace==True,regex=True) 
```

```python
import pandas as pd

df = pd.DataFrame(['$40,000*','$40000 conditions attached'], columns=['P'])
print(df)
#                             P
# 0                    $40,000*
# 1  $40000 conditions attached

df['P'] = df['P'].str.replace(r'\D+', '').astype('int')
print(df)
#yields
       P
0  40000
1  40000
```


## Regular expression to extract numbers from a string
```
^     # start of string
\s*   # optional whitespace
(\w+) # one or more alphanumeric characters, capture the match
\s*   # optional whitespace
\(    # a (
\s*   # optional whitespace
(\d+) # a number, capture the match
\D+   # one or more non-digits
(\d+) # a number, capture the match
\D+   # one or more non-digits
\)    # a )
\s*   # optional whitespace
$     # end of string

[^0-9]+([0-9]+)[^0-9]+([0-9]+).+
```

## Delete column from pandas DataFrame 
```python
del df['column_name']
```


## Data Interpolation
```python
from scipy import interpolate
from scipy.optimize import fsolve
import math

x = np.array([10,20,30,40,50])
y = np.array([0.2,0.6,-0.2,-0.5,0.7])

tck = interpolate.splrep(x, y, s=0)

xnew = np.arange(10,50,1)
ynew = interpolate.splev(xnew, tck, der=0)
# ynewder1 = interpolate.splev(xnew, tck, der=1)
# ynewder2 = interpolate.splev(xnew, tck, der=2)

plt.scatter(xnew,ynew)

```

## How can I replace all the NaN values with Zero's in a column of a pandas dataframe
```python
df[1].fillna(0, inplace=True)
```

## How to add an empty column to a dataframe?
```python
df = pd.DataFrame({"A": [1,2,3], "B": [2,3,4]})
df
Out[18]:
   A  B
0  1  2
1  2  3
2  3  4

df.assign(C="",D=np.nan)
Out[21]:
   A  B C   D
0  1  2   NaN
1  2  3   NaN
2  3  4   NaN
```

## Pandas add one day to column
```python
montdist['date'] + pd.DateOffset(1)
pd.DatetimeIndex(df.date) + pd.offsets.Hour(1)
mondist['shifted_date']=mondist.date + datetime.timedelta(days=1)
df['newdate'] = pd.to_datetime(df['date']).apply(pd.DateOffset(1))
df['newdate'] = pd.Series(index=df.index).tshift(periods=1, freq='D').index
```

## Pandas: Convert Timestamp to datetime.date
```python
In [11]: t = pd.Timestamp('2013-12-25 00:00:00')

In [12]: t.date()
Out[12]: datetime.date(2013, 12, 25)

In [13]: t.date() == datetime.date(2013, 12, 25)
Out[13]: True
```

# datetime to string with series in python pandas
```python
date = dataframe.index #date is the datetime index
date = dates.strftime('%Y-%m-%d') #this will return you a numpy array, element is string.
dstr = date.tolist()
```

# Python NumPy: Get the values and indices of the elements that are bigger than 10 in a given array

- ![](https://www.w3resource.com/w3r_images/python-numpy-image-exercise-31.png){:height="30%" width="30%"}

```python
import numpy as np
x = np.array([[0, 10, 20], [20, 30, 40]])
print("Original array: ")
print(x)
print("Values bigger than 10 =", x[x>10])
print("Their indices are ", np.nonzero(x > 10))

Original array:                                                        
[[ 0 10 20]                                                            
 [20 30 40]]                                                           
Values bigger than 10 = [20 20 30 40]                                  
Their indices are  (array([0, 1, 1, 1]), array([2, 0, 1, 2]))
```


# Reference 
- [Accessing pandas dataframe columns, rows, and cells](https://pythonhow.com/accessing-dataframe-columns-rows-and-cells/)

- [How To Change Column Names and Row Indexes Simultaneously in Pandas](http://cmdlinetips.com/2018/03/how-to-change-column-names-and-row-indexes-in-pandas/)

- [Changing a specific column name in pandas DataFrame](https://stackoverflow.com/questions/20868394/changing-a-specific-column-name-in-pandas-dataframe)

- [Create DateTimeIndex in Pandas](https://stackoverflow.com/questions/36506149/create-datetimeindex-in-pandas)

- [How to extract specific content in a pandas dataframe with a regex?](https://stackoverflow.com/questions/36028932/how-to-extract-specific-content-in-a-pandas-dataframe-with-a-regex)

- [Regular expression to extract numbers from a string](https://stackoverflow.com/questions/4187356/regular-expression-to-extract-numbers-from-a-string)


- [Delete column from pandas DataFrame using del df.column_name](https://stackoverflow.com/questions/13411544/delete-column-from-pandas-dataframe-using-del-df-column-name)