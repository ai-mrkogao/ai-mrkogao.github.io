---
title: "11 Experiments (reward environment design)"
date: 2018-08-21
classes: wide
use_math: true
tags: economic index python stock utils kospi keras tensorflow reinforcement_learning
category: stock
---

## Introduction
- Sector Selection from more than 2000 stock indices 
- I will use the Q-Learning with DQN (replay memory)
- I will design reward environments for each stocks 

## Retrieving each stock historical data
```python
dftotalname = pd.read_pickel("...")
dftotalname.shape
>>(2218,11)
```


- ## sort the data with capital size
```python
s = lambda f: f.replace(',','')
dftotalname['자본금(원)'] = dftotalname['자본금(원)'].apply(s).astype(int)
dftotalname = dftotalname.sort_values(['자본금(원)'],ascending=False)
dftotalname = dftotalname.reset_index(drop=True)
```

- ## request the historical data from the server
  - more than 2000 stocks gathered

## RDP turning point simulation
- apply RDP points in the long term with 0.2 epsilon
- apply curvature values in the short term 
- RDP could find the turning points but reward values sometimes are not suitable


