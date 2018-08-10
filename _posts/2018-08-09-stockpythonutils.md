---
title: "Stock Python Utils"
date: 2018-08-09
classes: wide
use_math: true
tags: economic index python stock utils kospi
category: stock
---

# read yahoo data from datareader 
[](http://nukeguys.tistory.com/194)

```python
pip install fix_yahoo_finance

from pandas_datareader import data as pdr 

import fix_yahoo_finance as yf 

yf.pdr_override()
data = pdr.get_data_yahoo("SPY",start="2017-01-01",end="2017-04-30")
data = pdr.get_data_yahoo(["SPY","IWM"],start="2017-01-01",end="2017-04-30")
```


# Receive Kospi data using data-reader from Yahoo 
[](https://github.com/Data-plus/get_Kospi)


```python
import pandas_datareader as wb
	import pandas as pd
	import datetime
	import matplotlib.pyplot as plt
	
	pd.set_option('precision', 3)
	
	start = datetime.datetime(2014, 12, 8)
	end = datetime.datetime(2017, 11, 7)
	df_null = wb.DataReader("003490.KS","yahoo",start,end)
	df = df_null.dropna()
	
	kospi_chart = df.Close.plot(style='b')
	kospi_chart.set_title("Korean Air")
	kospi_chart.set_ylabel("Price")
	kospi_chart.set_xlim(str(start), str(end))
	
	print(df)
	print("Close Median", df['Close'].median())
	print()
	print(df['Close'].describe())
	print()
	print(df.corr())
	print()
	
	original_price = df.iloc[0,3]
	nextday_price = df.iloc[1,3]
	current_price = df.iloc[-1,3]
	month_after_price = df.iloc[30,3]
	
	nextday_change = ((nextday_price-original_price) / original_price)
	month_change = ((month_after_price-original_price) / original_price)
	current_change = ((current_price-original_price) / original_price)
	
	print("Next day Change: {:.2%}.".format(nextday_change))
	print("Month After Change: {:.2%}.".format(month_change))
	print("Current Change: {:.2%}.".format(current_change))
	
	plt.show()
```

# Financial price data reader (an alternative to google finance and yahoo finance in pandas-datareader) 
[](https://github.com/FinanceData/FinanceDataReader)

## Install

```bash
pip install finance-datareader # for install
pip install -U finance-datareader # for update
```
## Usage

```python
import FinanceDataReader as fdr

# Apple(AAPL), 2017~Now
df = fdr.DataReader('AAPL', '2017')

# AMAZON(AMZN), 2010~Now
df = fdr.DataReader('AMZN', '2010')

# Celltrion(068270), 2018-07-01~Now
df = fdr.DataReader('068270', '2018-07-01')

# country code: ex) 000150: Doosan(KR), Yihua Healthcare(CN)
df = fdr.DataReader('000150', '2018-01-01', '2018-03-30') # default: 'KR' 
df = fdr.DataReader('000150', '2018-01-01', '2018-03-30', country='KR')
df = fdr.DataReader('000150', '2018-01-01', '2018-03-30', country='CN')

# KOSPI index, 2015~Now
df = fdr.DataReader('KS11', '2015')

# Dow Jones Industrial(DJI), 2015년~Now
df = fdr.DataReader('DJI', '2015')

# USD/KRW, 1995~Now
df = fdr.DataReader('USD/KRW', '1995')

# Bitcoin KRW price (Bithumbs), 2016 ~ Now
df = fdr.DataReader('BTC/KRW', '2016')

# KRX stock symbols and names
df_krx = fdr.StockListing('KRX')

# S&P 500 symbols
df_spx = fdr.StockListing('S&P500')
```
