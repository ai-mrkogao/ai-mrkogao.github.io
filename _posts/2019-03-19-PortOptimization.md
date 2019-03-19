---
title: "Portfolio Optimization"
date: 2019-03-19
classes: wide
use_math: true
tags: math arima ar ma adf minimize optimization portfolio
category: finance
---


## Montecarlo Optimization  

```python
stocks = pd.concat([aapl,cisco,ibm,amzn],axis=1)
stocks.columns = ['aapl','cisco','ibm','amzn']

mean_daily_ret = stocks.pct_change(1).mean()
mean_daily_ret

stocks.pct_change(1).corr()


```

### set the log return  
```python
log_ret = np.log(stocks/stocks.shift(1))
log_ret.head()
```

### monte carlo simulation  
```python
num_ports = 15000

all_weights = np.zeros((num_ports,len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for ind in range(num_ports):

    # Create Random Weights
    weights = np.array(np.random.random(4))

    # Rebalance Weights
    weights = weights / np.sum(weights)
    
    # Save Weights
    all_weights[ind,:] = weights

    # Expected Return
    ret_arr[ind] = np.sum((log_ret.mean() * weights) *252)

    # Expected Variance
    vol_arr[ind] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))

    # Sharpe Ratio
    sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]
```

### find max sharp ration  
```python
sharpe_arr.argmax()
```    


## scipy optimization function  
```python
from scipy.optimize import minimize


def get_ret_vol_sr(weights):
    """
    Takes in weights, returns array or return,volatility, sharpe ratio
    """
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])

```

 
### target function  
```python
def neg_sharpe(weights):
    return  get_ret_vol_sr(weights)[2] * -1


# Sequential Least SQuares Programming (SLSQP).
opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)



```    