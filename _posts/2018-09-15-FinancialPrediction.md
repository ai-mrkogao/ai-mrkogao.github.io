---
title: "Financial Prediction"
date: 2018-09-15
classes: wide
use_math: true
tags: python keras tensorflow reinforcement_learning turi machine_learning platform image caption financial MACD Stochastic ohlc candlestick
category: reinforcement learning
---



## NY News Json download
```python
items = requests.get(url)
    
try:
    data = items.json()

    with open(file_str, 'w') as f:
        json.dump(data, f)
except:
    pass
```

## MACD
```python
def MACD(df, period1, period2, periodSignal):
    EMA1 = pd.DataFrame.ewm(df,span=period1).mean() # Provides exponential weighted functions
    EMA2 = pd.DataFrame.ewm(df,span=period2).mean() 
    
    MACD = EMA1-EMA2
    Signal = pd.DataFrame.ewm(MACD,periodSignal).mean()
    
    Histogram = MACD-Signal
    return Histogram
```

## Stochastic
```python
def stochastics_oscillator(df,period):
    l, h = pd.DataFrame.rolling(df, period).min(), pd.DataFrame.rolling(df, period).max()
    k = 100 * (df - l) / (h - l)
    return k
```

## ATR
```python
def ATR(df,period):
    '''
    Method A: Current High less the current Low
    '''
    df['H-L'] = abs(df['High']-df['Low'])
    df['H-PC'] = abs(df['High']-df['Close'].shift(1))
    df['L-PC'] = abs(df['Low']-df['Close'].shift(1))
    TR = df[['H-L','H-PC','L-PC']].max(axis=1)
    return TR.to_frame()
```

## OHLC plot
```python
%matplotlib notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import datetime as dt


df= pd.read_pickle('../histdata/KQ11from2001') # 055550_신한지주 088350_한화생명
#Reset the index to remove Date column from index
df_ohlc = df.reset_index()
df_ohlc = df_ohlc[["Date","Open","High",'Low',"Close"]]
df_ohlc = df_ohlc[-100:]
#Naming columns
df_ohlc.columns = ["Date","Open","High",'Low',"Close"]

#Converting dates column to float values
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

#Making plot
fig = plt.figure()
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=6, colspan=1)

#Converts raw mdate numbers to dates
ax1.xaxis_date()
plt.xlabel("Date")
# print(df_ohlc)

#Making candlestick plot
candlestick_ohlc(ax1,df_ohlc.values,width=1, colorup='r', colordown='b',alpha=0.75)
plt.ylabel("Price")
plt.legend()

plt.show()
```