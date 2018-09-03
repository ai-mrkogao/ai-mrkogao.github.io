---
title: "Stock Python Utils"
date: 2018-08-09
classes: wide
use_math: true
tags: economic index python stock utils kospi
category: stock
---

# read yahoo data from datareader 
[read yahoo data from datareader](http://nukeguys.tistory.com/194)

```python
pip install fix_yahoo_finance

from pandas_datareader import data as pdr 

import fix_yahoo_finance as yf 

yf.pdr_override()
data = pdr.get_data_yahoo("SPY",start="2017-01-01",end="2017-04-30")
data = pdr.get_data_yahoo(["SPY","IWM"],start="2017-01-01",end="2017-04-30")
```


# Receive Kospi data using data-reader from Yahoo 
[Receive Kospi data using data-reader from Yahoo](https://github.com/Data-plus/get_Kospi)


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
[Financial price data reader](https://github.com/FinanceData/FinanceDataReader)

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


## 거래소 상장회사검색 
[거래소 상장회사검색](http://marketdata.krx.co.kr/contents/MKD/04/0406/04060100/MKD04060100.jsp)

## 네이버재무제표 가져오기
[네이버재무제표 가져오기](https://gist.github.com/KimMyungSam?page=2)

## KRX 종목 마스터 만들기 
[KRX 종목 마스터 만들기 ](https://gist.github.com/KimMyungSam?page=2)

## 종목마스터+만들기mysql과+sqlalchemy.ipynb
[종목마스터+만들기mysql과+sqlalchemy.ipynb](https://gist.github.com/KimMyungSam?page=2)

## mysql install
```bash
pip3 install mysqlclient
sudo apt-get install python3-dev libmysqlclient-dev
pip3 install mysql-connector
pip3 install configparser
sudo apt-get install -y python3-mysqldb
sudo apt-get install libmysqlclient-dev
sudo pip3 install mysqlclient
pip3 install MySQL-python

sudo apt-get install python3-mysql.connector # import mysql 
sudo pip3 install sqlalchemy
```


## distutils install problem
[Upgrading to pip 10: It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall](https://github.com/pypa/pip/issues/5247)

Reduced version,
```bash
pip install --upgrade --force-reinstall pip==9.0.3
#Tried to re-install package
pip install xxx --disable-pip-version-check
#At last, recover the latest version for pip
pip install --upgrade pip
```


## The Trading Economics Application Programming Interface (API)
[The Trading Economics Application Programming Interface (API)](http://docs.tradingeconomics.com/#introduction)

```python
pip install tradingeconomics
import tradingeconomics as te
te.login('Your_Key:Your_Secret')
te.login()
te.getIndicatorData(country=['united states', 'china'], output_type='df')

Output:
           Country         ...             PreviousValueDate
0    United States         ...           2018-04-30T00:00:00
1    United States         ...           2018-06-15T00:00:00
2    United States         ...           2017-12-31T00:00:00
               ...         ...                           ...
314          China         ...           2016-12-31T00:00:00
315  United States         ...           2016-12-31T00:00:00
316  United States         ...           2018-04-30T00:00:00
317  United States         ...           2018-04-30T00:00:00

```

## yahoo finance
```python
pip install yahoo-finance
from yahoo_finance import Share
import pandas as pd
from pandas import DataFrame, Series
samsung = Share('005930.KS') 
df = DataFrame(samsung.get_historical('2016-07-04','2016-07-08'))

```

## google finance
```python
from googlefinance.get import get_code
# get_code('NASDAQ')
get_code('KOSPI')

Code 	Code_int 	Google 	Name
0 	001040 	1040 	KRX:001040 	CJ
1 	012630 	12630 	KRX:012630 	HDC
2 	082740 	82740 	KRX:082740 	HSD엔진
```

```python

Getting Historical Financial Data

Getting the Only Single Company’s Historical Financial Data

    code = ‘NASDAQ: code list’
    period = ‘30d’: 30 days (default), ‘1M’ : Month , ‘1Y’ : year
    interval = 86400 : 1 day (default), 60 * integer (seconds)

>>> from googlefinance.get import get_datum
>>> df = get_datum('KRX:005930', period='2M'， interval =86400)
date        Open     High     Low      Close    Volume
2018-05-04  53000.0  53900.0  51800.0  51900.0  39290305
2018-05-08  52600.0  53200.0  51900.0  52600.0  22907823
2018-05-09  52600.0  52800.0  50900.0  50900.0  15914664

```


```python
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like

from pandas_datareader import data
import fix_yahoo_finance as yf
yf.pdr_override()

start_date = '1996-05-06' #startdate를 1996년으로 설정해두면 가장 오래된 데이터부터 전부 가져올 수 있다.
tickers = ['067160.KQ', '035420.KS'] #1 아프리카tv와 네이버의 ticker(종목코드)
afreeca = data.get_data_yahoo(tickers[0], start_date)
naver = data.get_data_yahoo(tickers[1], start_date)
skhynix = data.get_data_yahoo('000660.KS', start_date)
kospi = data.get_data_yahoo('^KS11', start_date)
```


[파이썬으로 주식 데이터 가져오기](https://gomjellie.github.io/%ED%8C%8C%EC%9D%B4%EC%8D%AC/pandas/%EC%A3%BC%EC%8B%9D/2017/06/09/pandas-datareader-stock.html)